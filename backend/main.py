"""
FastAPI Backend for AI Data Analysis Assistant
With DeepSeek LLM Integration + Chart Explanation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from utils.helpers import get_dataset_summary
import pandas as pd
import numpy as np
import os
import io
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import requests
from dotenv import load_dotenv
from analysis import DataAnalyzer
from visualization import ChartGenerator

# Load environment variables
load_dotenv()

# ================================================================
# JSON SERIALIZATION HELPER
# ================================================================

def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ================================================================
# DEEPSEEK LLM SETUP
# ================================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_AVAILABLE = False
LLM_PROVIDER = "None"

if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY.startswith("sk-"):
    LLM_AVAILABLE = True
    LLM_PROVIDER = "DeepSeek"
    print(f"✅ DeepSeek API loaded successfully!")
else:
    print("⚠️ DEEPSEEK_API_KEY not found. Set it in .env file or Hugging Face Secrets.")

def call_llm(prompt, max_tokens=1200):
    """Call DeepSeek API.

    FIXED: max_tokens was previously hardcoded at 1200 for every call site,
    including the comprehensive 7-section /explain-data report, which caused
    it to get cut off mid-sentence before reaching Key Insights, Statistical
    Highlights, Use Cases, or the final Summary. Now callers can request a
    larger budget for longer structured reports.
    """
    if not LLM_AVAILABLE:
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"DeepSeek API Error: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"DeepSeek Error: {e}")
        return None

# ================================================================
# FASTAPI APP
# ================================================================

app = FastAPI(title="AI Data Analysis Assistant", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
analyzer = DataAnalyzer()
chart_path = None

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)

# ================================================================
# Pydantic Models
# ================================================================

class QuestionRequest(BaseModel):
    question: str

class ChartExplanationRequest(BaseModel):
    chart_type: str = "Auto"
    x_column: Optional[str] = None
    y_column: Optional[str] = None
# ================================================================
# API Endpoints
# ================================================================

@app.get("/")
async def root():
    return {
        "message": "AI Data Analysis Assistant API",
        "status": "running",
        "version": "1.0.0",
        "llm_available": LLM_AVAILABLE,
        "llm_provider": LLM_PROVIDER,
        "endpoints": ["/upload", "/summary", "/info", "/ask", "/chart", "/export-pdf", "/explain-chart"]
    }

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and analyze CSV file"""
    global analyzer, chart_path
    
    try:
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        df = pd.read_csv(file_path)
        analyzer.load_data(df)
        
        chart_gen = ChartGenerator(df)
        chart_path = chart_gen.generate_chart()
        
        summary = analyzer.get_summary()
        serializable_summary = convert_to_serializable(summary)
        
        return JSONResponse({
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully",
            "summary": serializable_summary,
            "llm_available": LLM_AVAILABLE,
            "llm_provider": LLM_PROVIDER
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply-cleaned-data")
async def apply_cleaned_data(file: UploadFile = File(...)):
    """Make an uploaded (frontend-cleaned) CSV the ACTIVE dataset for all
    analysis - Q&A, AI Explain, Statistics & Insights, and Charts - not just
    PDF export.

    ADDED: previously cleaning only lived in the Streamlit frontend's session
    state and was never sent back to the backend except as a one-off file
    for /export-pdf, so every other feature kept answering from the
    original uncleaned data even after the user cleaned it (e.g. reporting
    the original row count instead of the post-cleaning row count).
    """
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        analyzer.set_active_data(df)

        summary = analyzer.get_summary()
        serializable_summary = convert_to_serializable(summary)

        return JSONResponse({
            "success": True,
            "message": f"Cleaned dataset is now active: {len(df):,} rows",
            "summary": serializable_summary,
            "is_cleaned": True
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset-to-original")
async def reset_to_original():
    """Revert the active dataset back to the originally uploaded data,
    so Q&A/AI Explain/Insights/Charts go back to reporting on the
    original (uncleaned) row count and values."""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")

    success = analyzer.reset_to_original()
    if not success:
        raise HTTPException(status_code=400, detail="No original data available to reset to")

    summary = analyzer.get_summary()
    serializable_summary = convert_to_serializable(summary)

    return JSONResponse({
        "success": True,
        "message": f"Reverted to original dataset: {len(analyzer.df):,} rows",
        "summary": serializable_summary,
        "is_cleaned": False
    })

@app.get("/summary")
async def get_summary():
    """Get dataset summary"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    summary = analyzer.get_summary()
    serializable_summary = convert_to_serializable(summary)
    return JSONResponse({"success": True, "summary": serializable_summary})

@app.get("/info")
async def get_info():
    """Get basic dataset information"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    info = analyzer.get_basic_info()
    serializable_info = convert_to_serializable(info)
    return JSONResponse({"success": True, "info": serializable_info})

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a natural language question"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = analyzer.answer_question(request.question, call_llm)
    result["llm_available"] = LLM_AVAILABLE
    result["llm_provider"] = LLM_PROVIDER
    return JSONResponse(result)

@app.get("/full-dataset")
async def get_full_dataset():
    """Get the full dataset for preview"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        df_dict = analyzer.df.to_dict('records')
        columns = analyzer.df.columns.tolist()
        
        return JSONResponse({
            "success": True,
            "data": df_dict,
            "columns": columns,
            "rows": len(df_dict)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/original-dataset")
async def get_original_dataset():
    """Get the ORIGINAL (pre-cleaning) dataset, regardless of what is
    currently active.

    ADDED: /full-dataset always returns the currently ACTIVE dataset, which
    becomes the cleaned data after /apply-cleaned-data is called. The Data
    Cleaning tab's "Original Rows" vs "Cleaned Rows" comparison needs the
    TRUE original for this to be meaningful - without this endpoint, that
    comparison was comparing the cleaned data against itself, making real
    cleaning (e.g. duplicates actually removed) look like "no rows removed".
    """
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        original_df = analyzer.original_df if analyzer.original_df is not None else analyzer.df
        df_dict = original_df.to_dict('records')
        columns = original_df.columns.tolist()

        return JSONResponse({
            "success": True,
            "data": df_dict,
            "columns": columns,
            "rows": len(df_dict)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chart")
async def get_chart():
    """Generate and return a chart"""
    global chart_path
    
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    chart_gen = ChartGenerator(analyzer.df)
    chart_path = chart_gen.generate_chart()
    return FileResponse(chart_path, media_type="image/png")

@app.get("/chart/{chart_type}")
async def get_specific_chart(
    chart_type: str,
    x_column: str = None,
    y_column: str = None
):
    """Generate specific chart type"""
    global chart_path
    
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    chart_gen = ChartGenerator(analyzer.df)
    
    if chart_type == "bar":
        if not x_column:
            x_column = analyzer.df.columns[0]
        chart_path = chart_gen.generate_bar_chart(x_column, y_column)
    
    elif chart_type == "pie":
        if not x_column:
            categorical_cols = analyzer.df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                x_column = categorical_cols[0]
            else:
                x_column = analyzer.df.columns[0]
        chart_path = chart_gen.generate_pie_chart(x_column)
    
    elif chart_type == "histogram":
        if not x_column:
            numeric_cols = analyzer.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                x_column = numeric_cols[0]
            else:
                x_column = analyzer.df.columns[0]
        chart_path = chart_gen.generate_histogram(x_column)
    
    elif chart_type == "scatter":
        numeric_cols = analyzer.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for scatter plot")
        chart_path = chart_gen.generate_scatter(numeric_cols[0], numeric_cols[1])
    
    else:
        chart_path = chart_gen.generate_chart()
    
    return FileResponse(chart_path, media_type="image/png")

@app.post("/generate-chart")
async def generate_chart(request: dict):
    """Generate chart based on user selection"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        chart_type = request.get('chart_type', 'bar')
        x_column = request.get('x_column')
        y_column = request.get('y_column')  # This can be None for Pie/Histogram
        title = request.get('title')
        color_palette = request.get('color_palette', 'viridis')
        
        df = analyzer.df
        
        # ============================================================
        # HANDLE DIFFERENT CHART TYPES
        # ============================================================
        
        # For Pie chart - y_column is not needed
        if chart_type == 'pie':
            if not x_column:
                x_column = df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else df.columns[0]
            chart_path = generate_pie_chart(df, x_column, title, color_palette)
        
        # For Histogram - y_column is not needed
        elif chart_type == 'histogram':
            if not x_column:
                x_column = df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else df.columns[0]
            chart_path = generate_histogram_chart(df, x_column, title, color_palette)
        
        # For Boxplot - FIXED: pass BOTH x_column (category to group by) and
        # y_column (numeric value) so a proper grouped boxplot is generated
        # instead of only ever plotting x_column.
        elif chart_type == 'boxplot':
            chart_path = generate_boxplot_chart(df, x_column, y_column, title, color_palette)
        
        # For Bar chart - needs x and y
        elif chart_type == 'bar':
            if not x_column:
                x_column = df.columns[0]
            if not y_column:
                y_column = df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else df.columns[0]
            chart_path = generate_bar_chart(df, x_column, y_column, title, color_palette)
        
        # For Scatter plot - needs x and y
        elif chart_type == 'scatter':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not x_column:
                x_column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
            if not y_column:
                y_column = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            chart_path = generate_scatter_chart(df, x_column, y_column, title, color_palette)
        
        # For Line chart - needs x and y
        elif chart_type == 'line':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not x_column:
                x_column = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
            if not y_column:
                y_column = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            chart_path = generate_line_chart(df, x_column, y_column, title, color_palette)
        
        # For Heatmap - no x or y needed
        elif chart_type == 'heatmap':
            chart_path = generate_heatmap_chart(df, title, color_palette)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported chart type: {chart_type}")
        
        return FileResponse(chart_path, media_type="image/png")
    
    except Exception as e:
        print(f"Chart generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# CHART GENERATOR FUNCTIONS
# ================================================================

def generate_bar_chart(df, x_col, y_col, title=None, palette="viridis"):
    """Generate bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Aggregate data
    data = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(15)
    
    # FIX: Use plt.colormaps.get_cmap() instead of plt.cm.get_cmap()
    cmap = plt.colormaps.get_cmap(palette)
    colors = cmap(np.linspace(0.2, 0.8, len(data)))
    
    bars = ax.bar(data.index.astype(str), data.values, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_title(title or f"{y_col} by {x_col}", fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars, data.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:,.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    filename = f"charts/bar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def generate_pie_chart(df, col, title=None, palette="Set3"):
    """Generate pie chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = df[col].value_counts().head(8)
    
    # FIX: Use plt.colormaps.get_cmap() instead of plt.cm.get_cmap()
    cmap = plt.colormaps.get_cmap(palette)
    colors = cmap(np.linspace(0, 1, len(data)))
    
    wedges, texts, autotexts = ax.pie(
        data.values, 
        labels=data.index.astype(str), 
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0.02] * len(data)
    )
    
    ax.set_title(title or f"Distribution of {col}", fontsize=14, fontweight='bold')
    
    for text in texts:
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    filename = f"charts/pie_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def generate_histogram_chart(df, col, title=None, palette="viridis"):
    """Generate histogram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(df[col].dropna(), bins=20, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # FIX: Use plt.colormaps.get_cmap() instead of plt.cm.get_cmap()
    cmap = plt.colormaps.get_cmap(palette)
    colors = cmap(np.linspace(0.2, 0.8, len(patches)))
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    ax.axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df[col].mean():.2f}')
    ax.axvline(df[col].median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {df[col].median():.2f}')
    
    ax.set_title(title or f"Distribution of {col}", fontsize=14, fontweight='bold')
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    
    filename = f"charts/histogram_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def generate_scatter_chart(df, x_col, y_col, title=None, palette="viridis"):
    """Generate scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    color_col = None
    for col in numeric_cols:
        if col not in [x_col, y_col]:
            color_col = col
            break
    
    if color_col:
        scatter = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap=palette, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
        plt.colorbar(scatter, label=color_col)
    else:
        ax.scatter(df[x_col], df[y_col], color='#2E86AB', alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
    
    ax.set_title(title or f"{y_col} vs {x_col}", fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"charts/scatter_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def generate_line_chart(df, x_col, y_col, title=None, palette="viridis"):
    """Generate line chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if pd.api.types.is_numeric_dtype(df[x_col]):
        sorted_df = df.sort_values(x_col)
        ax.plot(sorted_df[x_col], sorted_df[y_col], 'o-', color='#2E86AB', linewidth=2, markersize=6)
    else:
        ax.plot(df[x_col], df[y_col], 'o-', color='#2E86AB', linewidth=2, markersize=6)
    
    ax.fill_between(df[x_col], df[y_col], alpha=0.1)
    ax.set_title(title or f"{y_col} vs {x_col} (Line Chart)", fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"charts/line_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def generate_boxplot_chart(df, x_column=None, y_column=None, title=None, palette="viridis"):
    """Generate boxplot.

    FIXED: previously this only ever accepted a single column, so giving both
    an X (category) and Y (numeric) column silently dropped the Y column and
    only plotted X. Now, when both a category column (x_column) and a numeric
    column (y_column) are provided, a proper grouped boxplot is generated
    (y_column's distribution split out per x_column category). Falls back to
    a single-column boxplot, or a boxplot of all numeric columns, if only one
    or neither is given.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    has_x = x_column and x_column in df.columns
    has_y = y_column and y_column in df.columns

    if has_x and has_y and x_column != y_column:
        df.boxplot(column=y_column, by=x_column, ax=ax)
        ax.set_title(title or f"Boxplot of {y_column} by {x_column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(x_column, fontsize=12)
        ax.set_ylabel(y_column, fontsize=12)
        plt.suptitle('')  # remove pandas' auto-added "Boxplot grouped by ..." subtitle
        plt.xticks(rotation=45, ha='right')
    elif has_y:
        df.boxplot(column=y_column, ax=ax)
        ax.set_title(title or f"Boxplot of {y_column}", fontsize=14, fontweight='bold')
        ax.set_ylabel(y_column, fontsize=12)
    elif has_x:
        df.boxplot(column=x_column, ax=ax)
        ax.set_title(title or f"Boxplot of {x_column}", fontsize=14, fontweight='bold')
        ax.set_ylabel(x_column, fontsize=12)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols].boxplot(ax=ax)
        ax.set_title(title or "Boxplot of Numeric Columns", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    filename = f"charts/boxplot_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def generate_heatmap_chart(df, title=None, palette="coolwarm"):
    """Generate correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return generate_histogram_chart(df, numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0], title)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        fmt='.2f', 
        cmap=palette,
        square=True, 
        linewidths=0.5, 
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    
    ax.set_title(title or "Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f"charts/heatmap_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def build_chart_context(df, chart_type, x_col, y_col, numeric_cols, categorical_cols, summary):
    """Build a data context string describing what a given chart shows.
    Used to feed the AI explanation prompt for both the /explain-chart endpoint
    and the PDF export, so every chart type (Pie, Histogram, Boxplot, Scatter,
    Line, Heatmap, Bar, Auto) gets real data context even when only one of
    x_col/y_col is provided.
    """
    chart_context = ""
    
    # ============================================================
    # PIE CHART (FIXED)
    # ============================================================
    if chart_type == "Pie":
        if x_col:
            # Get the actual data for the pie chart
            value_counts = df[x_col].value_counts()
            total = len(df)
            
            chart_context = f"""
CHART TYPE: Pie Chart
CATEGORY COLUMN: {x_col}
TOTAL RECORDS: {len(df):,}

DISTRIBUTION (Top categories):
"""
            for cat, count in value_counts.head(8).items():
                pct = (count / total) * 100
                chart_context += f"- {cat}: {count:,} ({pct:.1f}%)\n"
            
            # Add summary
            top_category = value_counts.index[0]
            top_count = value_counts.iloc[0]
            top_pct = (top_count / total) * 100
            chart_context += f"\nThe largest category is '{top_category}' with {top_count:,} occurrences ({top_pct:.1f}% of the data)."
            
        elif categorical_cols:
            cat_col = categorical_cols[0]
            value_counts = df[cat_col].value_counts()
            total = len(df)
            
            chart_context = f"""
CHART TYPE: Pie Chart
CATEGORY COLUMN: {cat_col}
TOTAL RECORDS: {len(df):,}

DISTRIBUTION (Top categories):
"""
            for cat, count in value_counts.head(8).items():
                pct = (count / total) * 100
                chart_context += f"- {cat}: {count:,} ({pct:.1f}%)\n"
            
            top_category = value_counts.index[0]
            top_count = value_counts.iloc[0]
            top_pct = (top_count / total) * 100
            chart_context += f"\nThe largest category is '{top_category}' with {top_count:,} occurrences ({top_pct:.1f}% of the data)."
        else:
            chart_context = "No categorical column found for Pie Chart."

    # ============================================================
    # BAR CHART
    # ============================================================
    elif chart_type == "Bar":
        if x_col and y_col:
            top_5 = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(5)
            chart_context = f"""
CHART TYPE: Bar Chart
X-AXIS: {x_col}
Y-AXIS: {y_col}
TOTAL RECORDS: {len(df):,}

TOP 5 CATEGORIES:
"""
            for cat, val in top_5.items():
                chart_context += f"- {cat}: {val:,.0f}\n"
            
            top = top_5.index[0]
            top_val = top_5.values[0]
            chart_context += f"\nThe highest {y_col} is from '{top}' with {top_val:,.0f}."
        elif categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            top_5 = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(5)
            chart_context = f"""
CHART TYPE: Bar Chart
X-AXIS: {cat_col}
Y-AXIS: {num_col}
TOTAL RECORDS: {len(df):,}

TOP 5 CATEGORIES:
"""
            for cat, val in top_5.items():
                chart_context += f"- {cat}: {val:,.0f}\n"
            
            top = top_5.index[0]
            top_val = top_5.values[0]
            chart_context += f"\nThe highest {num_col} is from '{top}' with {top_val:,.0f}."

    # ============================================================
    # HISTOGRAM
    # ============================================================
    elif chart_type == "Histogram":
        if x_col:
            stats = summary.get('numeric_stats', {}).get(x_col, {})
            chart_context = f"""
CHART TYPE: Histogram
COLUMN: {x_col}
TOTAL RECORDS: {len(df):,}

KEY STATISTICS:
- Min: {stats.get('min', 'N/A')}
- Max: {stats.get('max', 'N/A')}
- Mean: {stats.get('mean', 'N/A'):.2f}
- Median: {stats.get('median', 'N/A'):.2f}
- Standard Deviation: {stats.get('std', 'N/A'):.2f}
"""
        elif numeric_cols:
            col = numeric_cols[0]
            stats = summary.get('numeric_stats', {}).get(col, {})
            chart_context = f"""
CHART TYPE: Histogram
COLUMN: {col}
TOTAL RECORDS: {len(df):,}

KEY STATISTICS:
- Min: {stats.get('min', 'N/A')}
- Max: {stats.get('max', 'N/A')}
- Mean: {stats.get('mean', 'N/A'):.2f}
- Median: {stats.get('median', 'N/A'):.2f}
"""

    # ============================================================
    # SCATTER PLOT
    # ============================================================
    elif chart_type == "Scatter":
        if x_col and y_col:
            chart_context = f"""
CHART TYPE: Scatter Plot
X-AXIS: {x_col}
Y-AXIS: {y_col}
TOTAL POINTS: {len(df):,}

STATISTICS:
- {x_col}: Min={df[x_col].min():.2f}, Max={df[x_col].max():.2f}, Mean={df[x_col].mean():.2f}
- {y_col}: Min={df[y_col].min():.2f}, Max={df[y_col].max():.2f}, Mean={df[y_col].mean():.2f}
"""
        elif len(numeric_cols) >= 2:
            col1 = numeric_cols[0]
            col2 = numeric_cols[1]
            chart_context = f"""
CHART TYPE: Scatter Plot
X-AXIS: {col1}
Y-AXIS: {col2}
TOTAL POINTS: {len(df):,}
"""

    # ============================================================
    # LINE CHART
    # ============================================================
    elif chart_type == "Line":
        if x_col and y_col:
            chart_context = f"""
CHART TYPE: Line Chart
X-AXIS: {x_col}
Y-AXIS: {y_col}
TOTAL POINTS: {len(df):,}
"""
        elif len(numeric_cols) >= 2:
            col1 = numeric_cols[0]
            col2 = numeric_cols[1]
            chart_context = f"""
CHART TYPE: Line Chart
X-AXIS: {col1}
Y-AXIS: {col2}
TOTAL POINTS: {len(df):,}
"""

    # ============================================================
    # BOXPLOT
    # ============================================================
    elif chart_type == "Boxplot":
        if x_col and y_col and x_col in df.columns and y_col in df.columns and x_col != y_col:
            # Grouped boxplot: x_col is the category to group by, y_col is the
            # numeric column being distributed. This matches generate_boxplot_chart's
            # df.boxplot(column=y_col, by=x_col) behavior, so the explanation
            # reflects both axes shown on the actual chart.
            grouped = df.groupby(x_col)[y_col]
            group_stats = grouped.describe()[['min', '25%', '50%', '75%', 'max', 'mean']]
            group_stats = group_stats.sort_values('50%', ascending=False).head(8)

            chart_context = f"""
CHART TYPE: Boxplot (grouped)
CATEGORY (X-AXIS): {x_col}
VALUE (Y-AXIS): {y_col}
TOTAL RECORDS: {len(df):,}
NUMBER OF GROUPS: {df[x_col].nunique()}

PER-GROUP STATISTICS (top groups by median, sorted descending):
"""
            for cat, row in group_stats.iterrows():
                chart_context += (
                    f"- {cat}: Min={row['min']:.2f}, Q1={row['25%']:.2f}, "
                    f"Median={row['50%']:.2f}, Q3={row['75%']:.2f}, "
                    f"Max={row['max']:.2f}, Mean={row['mean']:.2f}\n"
                )

            top_group = group_stats.index[0]
            top_median = group_stats.iloc[0]['50%']
            chart_context += f"\nThe group with the highest median {y_col} is '{top_group}' (median: {top_median:.2f})."

        elif y_col:
            stats = summary.get('numeric_stats', {}).get(y_col, {})
            q1 = df[y_col].quantile(0.25) if y_col in df.columns else stats.get('min', 'N/A')
            q3 = df[y_col].quantile(0.75) if y_col in df.columns else stats.get('max', 'N/A')
            chart_context = f"""
CHART TYPE: Boxplot
COLUMN: {y_col}
TOTAL RECORDS: {len(df):,}

KEY STATISTICS:
- Min: {stats.get('min', 'N/A')}
- Max: {stats.get('max', 'N/A')}
- Mean: {stats.get('mean', 'N/A'):.2f}
- Median: {stats.get('median', 'N/A'):.2f}
- Q1: {q1 if isinstance(q1, str) else f'{q1:.2f}'}
- Q3: {q3 if isinstance(q3, str) else f'{q3:.2f}'}
"""
        elif x_col:
            stats = summary.get('numeric_stats', {}).get(x_col, {})
            chart_context = f"""
CHART TYPE: Boxplot
COLUMN: {x_col}
TOTAL RECORDS: {len(df):,}

KEY STATISTICS:
- Min: {stats.get('min', 'N/A')}
- Max: {stats.get('max', 'N/A')}
- Mean: {stats.get('mean', 'N/A'):.2f}
- Median: {stats.get('median', 'N/A'):.2f}
"""

    # ============================================================
    # HEATMAP
    # ============================================================
    elif chart_type == "Heatmap":
        chart_context = f"""
CHART TYPE: Heatmap
NUMERIC COLUMNS: {', '.join(numeric_cols[:10])}
TOTAL RECORDS: {len(df):,}
"""

    # ============================================================
    # AUTO (Default)
    # ============================================================
    else:
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            top_5 = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(5)
            chart_context = f"""
CHART TYPE: Bar Chart (Auto-detected)
X-AXIS: {cat_col}
Y-AXIS: {num_col}
TOTAL RECORDS: {len(df):,}

TOP 5 CATEGORIES:
"""
            for cat, val in top_5.items():
                chart_context += f"- {cat}: {val:,.0f}\n"
        elif categorical_cols:
            cat_col = categorical_cols[0]
            value_counts = df[cat_col].value_counts()
            chart_context = f"""
CHART TYPE: Pie Chart (Auto-detected)
COLUMN: {cat_col}
TOTAL RECORDS: {len(df):,}

DISTRIBUTION:
"""
            for cat, count in value_counts.head(5).items():
                pct = (count / len(df)) * 100
                chart_context += f"- {cat}: {count} ({pct:.1f}%)\n"
        elif numeric_cols:
            col = numeric_cols[0]
            stats = summary.get('numeric_stats', {}).get(col, {})
            chart_context = f"""
CHART TYPE: Histogram (Auto-detected)
COLUMN: {col}
TOTAL RECORDS: {len(df):,}

KEY STATISTICS:
- Min: {stats.get('min', 'N/A')}
- Max: {stats.get('max', 'N/A')}
- Mean: {stats.get('mean', 'N/A'):.2f}
- Median: {stats.get('median', 'N/A'):.2f}
"""
        else:
            chart_context = "No suitable data found for chart."


    return chart_context

@app.post("/explain-chart")
async def explain_chart(request: ChartExplanationRequest = None):
    """Generate AI explanation for the current chart"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    if not LLM_AVAILABLE:
        return JSONResponse({
            "success": False,
            "explanation": "⚠️ DeepSeek API key not configured.",
            "llm_available": False
        })
    
    try:
        chart_type = "Auto"
        x_col = None
        y_col = None
        
        if request:
            chart_type = request.chart_type
            x_col = request.x_column
            y_col = request.y_column
        
        df = analyzer.df
        summary = analyzer.get_summary()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # ============================================================
        # BUILD CHART CONTEXT BASED ON CHART TYPE
        # ============================================================
        
        chart_context = build_chart_context(df, chart_type, x_col, y_col, numeric_cols, categorical_cols, summary)


        # ============================================================
        # BUILD PROMPT
        # ============================================================
        
        prompt = f"""
You are a data analysis expert. Analyze this chart and explain what it shows.

{chart_context}

Please provide:
1. What this chart shows in simple terms (1-2 sentences)
2. The most prominent observation from this chart (1 sentence)
3. One key insight from this chart (1 sentence)

IMPORTANT: Use the exact column names: {x_col if x_col else 'N/A'}
ONLY talk about the data in this chart. Keep it short and focused.
"""
        
        llm_response = call_llm(prompt)
        
        if llm_response:
            return JSONResponse({
                "success": True,
                "explanation": llm_response,
                "llm_provider": LLM_PROVIDER
            })
        else:
            return JSONResponse({
                "success": False,
                "explanation": f"⚠️ Failed to generate explanation for {chart_type} chart.",
                "llm_available": False
            })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "explanation": f"Error: {str(e)}",
            "llm_available": False
        })

@app.post("/explain-data")
async def explain_data():
    """Generate comprehensive AI explanation for the entire dataset"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    if not LLM_AVAILABLE:
        return JSONResponse({
            "success": False,
            "explanation": "⚠️ DeepSeek API key not configured.",
            "llm_available": False
        })
    
    try:
        df = analyzer.df
        summary = analyzer.get_summary()
        
        # Basic info
        total_rows = summary.get('total_rows', 0)
        total_cols = summary.get('total_columns', 0)
        columns_info = summary.get('column_names', [])
        
        # Column details
        column_details = ""
        for col in columns_info:
            dtype = summary.get('data_types', {}).get(col, 'unknown')
            missing = summary.get('missing_values', {}).get(col, 0)
            missing_pct = summary.get('missing_percentage', {}).get(col, 0)
            column_details += f"- **{col}**: {dtype} (Missing: {missing} / {missing_pct:.1f}%)\n"
        
        # Numeric stats
        numeric_stats = summary.get('numeric_stats', {})
        stats_text = ""
        for col, stats in numeric_stats.items():
            stats_text += f"- **{col}**: Min={stats['min']:.2f}, Max={stats['max']:.2f}, Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Std={stats['std']:.2f}\n"
        
        # Categorical stats
        # FIXED: previously filtered summary['data_types'] for the literal
        # string 'object', which silently returns empty if that helper
        # stores dtypes differently - causing the AI to see zero categorical
        # context even on datasets with 9 categorical columns. Compute
        # directly from the dataframe instead, which is always correct.
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        cat_text = ""
        for col in categorical_cols[:5]:
            top_values = df[col].value_counts().head(3)
            cat_text += f"- **{col}**: {dict(top_values)}\n"
        
        # Data quality
        duplicates = df.duplicated().sum()
        missing_total = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        quality_score = ((total_cells - missing_total - duplicates) / total_cells) * 100
        
        # Sample data
        sample_data = df.head(5).to_string()
        
        prompt = f"""
        You are a senior data analyst. Provide a COMPREHENSIVE, DETAILED explanation of the following dataset.
        
        ============================================================
        DATASET OVERVIEW
        ============================================================
        Total Rows: {total_rows:,}
        Total Columns: {total_cols}
        Columns: {', '.join(columns_info)}
        
        ============================================================
        COLUMN DETAILS
        ============================================================
        {column_details}
        
        ============================================================
        NUMERIC COLUMNS STATISTICS
        ============================================================
        {stats_text if stats_text else 'No numeric columns found.'}
        
        ============================================================
        CATEGORICAL COLUMNS (Top 3 values)
        ============================================================
        {cat_text if cat_text else 'No categorical columns found.'}
        
        ============================================================
        DATA QUALITY
        ============================================================
        - Data Quality Score: {quality_score:.1f}%
        - Duplicate Rows: {duplicates:,}
        - Missing Values: {missing_total:,}
        - Total Cells: {total_cells:,}
        - Missing Percentage: {(missing_total / total_cells * 100):.2f}%
        
        ============================================================
        SAMPLE DATA (First 5 rows)
        ============================================================
        {sample_data}
        
        ============================================================
        ANALYSIS REQUIREMENTS
        ============================================================
        Please provide a COMPREHENSIVE analysis with the following sections:
        
        1. DATASET OVERVIEW (3-4 sentences):
           - What does this dataset represent?
           - What is the purpose of this data?
           - What are the main variables?
        
        2. COLUMN-BY-COLUMN ANALYSIS (for each column):
           - Explain what each column represents
           - What type of data it contains
           - Key observations from the data
        
        3. DATA QUALITY ASSESSMENT (2-3 bullet points):
           - Assess the completeness and cleanliness
           - Mention any issues (missing values, duplicates, outliers)
           - Suggestions for data cleaning
        
        4. KEY INSIGHTS (3-4 bullet points):
           - Most important patterns you observe
           - Relationships between columns
           - Surprising findings
        
        5. STATISTICAL HIGHLIGHTS (2-3 bullet points):
           - Key numbers from the data
           - Important averages, ranges, distributions
        
        6. POTENTIAL USE CASES (2-3 bullet points):
           - What could this data be used for?
           - What kind of analysis would be valuable?
           - What business questions could it answer?
        
        7. SUMMARY (2-3 sentences):
           - Simple summary for non-technical stakeholders
        
        Make it professional, comprehensive, and easy to understand.
        """
        
        llm_response = call_llm(prompt, max_tokens=2500)
        
        if llm_response:
            return JSONResponse({
                "success": True,
                "explanation": llm_response,
                "llm_provider": LLM_PROVIDER
            })
        else:
            return JSONResponse({
                "success": False,
                "explanation": "⚠️ Failed to generate explanation. Please check your DeepSeek API key.",
                "llm_available": False
            })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "explanation": f"Error: {str(e)}",
            "llm_available": False
        })

@app.post("/generate-insights")
async def generate_insights():
    """Generate statistics-focused AI insights for the Statistics & Insights tab.

    ADDED: previously this tab reused /explain-data, which is meant for the
    AI Explain tab and gives a column-by-column breakdown. This endpoint
    gives a distinct, statistics-focused structure instead: Data Overview,
    Data Statistics, Key Findings, Noticeable Patterns, Data Summary.
    """
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")

    if not LLM_AVAILABLE:
        return JSONResponse({
            "success": False,
            "explanation": "DeepSeek API key not configured.",
            "llm_available": False
        })

    try:
        df = analyzer.df
        summary = analyzer.get_summary()

        total_rows = summary.get('total_rows', 0)
        total_cols = summary.get('total_columns', 0)
        columns_info = summary.get('column_names', [])

        numeric_stats = summary.get('numeric_stats', {})
        stats_text = ""
        for col, stats in numeric_stats.items():
            stats_text += (
                f"- {col}: Min={stats['min']:.2f}, Max={stats['max']:.2f}, "
                f"Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Std={stats['std']:.2f}\n"
            )

        # FIXED: compute directly from df instead of trusting summary's dtype
        # string format, which was silently returning empty categorical_cols.
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        cat_text = ""
        for col in categorical_cols[:5]:
            top_values = df[col].value_counts().head(3)
            cat_text += f"- {col}: {dict(top_values)}\n"

        # Outlier summary (IQR method) so "Noticeable Patterns" can reference real numbers
        outlier_text = ""
        for col in numeric_stats.keys():
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_count = df[(df[col] < lower) | (df[col] > upper)].shape[0]
                if outlier_count > 0:
                    outlier_text += f"- {col}: {outlier_count:,} outliers ({(outlier_count/len(df)*100):.1f}%)\n"

        duplicates = df.duplicated().sum()
        missing_total = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        quality_score = ((total_cells - missing_total - duplicates) / total_cells) * 100

        prompt = f"""
You are a senior data analyst. Analyze the following dataset's STATISTICS and provide insights.

DATASET: {total_rows:,} rows, {total_cols} columns
COLUMNS: {', '.join(columns_info)}

NUMERIC STATISTICS:
{stats_text if stats_text else 'No numeric columns found.'}

CATEGORICAL TOP VALUES:
{cat_text if cat_text else 'No categorical columns found.'}

OUTLIERS DETECTED (IQR method):
{outlier_text if outlier_text else 'No significant outliers detected.'}

DATA QUALITY:
- Data Quality Score: {quality_score:.1f}%
- Duplicate Rows: {duplicates:,}
- Missing Values: {missing_total:,}

IMPORTANT: Do NOT use Markdown formatting (no #, no **, no bullet symbols like • or -). Write plain sentences.

Structure your response in EXACTLY these 5 sections, in this order, using these exact section headings in capital letters:

DATA OVERVIEW:
[2-3 sentences describing what this dataset is and its scale - rows, columns, numeric vs categorical mix]

DATA STATISTICS:
[3-4 sentences highlighting the most important numeric ranges, averages, and spreads from the statistics above]

KEY FINDINGS:
[3-4 sentences on the most important takeaways from the statistics - which columns show the widest spread, most skew, or most concentration]

NOTICEABLE PATTERNS:
[2-3 sentences on outliers, imbalances in categorical distributions, or unusual statistical patterns you observe]

DATA SUMMARY:
[2-3 sentences giving a simple, non-technical wrap-up of what the statistics tell us overall]
"""

        llm_response = call_llm(prompt, max_tokens=1800)

        if llm_response:
            return JSONResponse({
                "success": True,
                "explanation": llm_response,
                "llm_provider": LLM_PROVIDER
            })
        else:
            return JSONResponse({
                "success": False,
                "explanation": "Failed to generate insights. Please check your DeepSeek API key.",
                "llm_available": False
            })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "explanation": f"Error: {str(e)}",
            "llm_available": False
        })

@app.post("/export-pdf")
async def export_pdf(
    request: Request,
    chart_type: str = Form("Auto"),
    x_column: str = Form(None),
    y_column: str = Form(None),
    use_cleaned: str = Form("false"),
    chart: UploadFile = File(None),
    cleaned_data: UploadFile = File(None)
):
    """Export comprehensive analysis as PDF using current chart and data"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.lib.pagesizes import letter
        import io
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = f"analysis_report_{timestamp}.pdf"
        
        # ================================================================
        # USE CLEANED DATA IF PROVIDED
        # ================================================================
        
        is_cleaned = use_cleaned.lower() == "true"
        
        # Get the dataframe - use cleaned data if provided
        df = analyzer.df  # Default to original
        
        if cleaned_data and cleaned_data.filename:
            try:
                content = await cleaned_data.read()
                df = pd.read_csv(io.BytesIO(content))
                print(f"✅ Using CLEANED data: {len(df)} rows")
            except Exception as e:
                print(f"Error reading cleaned data: {e}")
                df = analyzer.df
                is_cleaned = False
        else:
            df = analyzer.df
        
        # Get summary from the data
        summary = analyzer.get_summary()
        
        # Get column info
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Calculate quality metrics
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        quality_score = ((total_cells - missing_cells - df.duplicated().sum()) / total_cells) * 100
        
        # ================================================================
        # HANDLE CHART
        # ================================================================
        
        chart_path = None
        
        if chart and chart.filename:
            chart_path = f"charts/user_chart_{timestamp}.png"
            with open(chart_path, "wb") as f:
                f.write(await chart.read())
        else:
            chart_gen = ChartGenerator(df)
            chart_path = chart_gen.generate_chart()
        
        # ================================================================
        # BUILD PDF
        # ================================================================
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # ================================================================
        # PAGE 1: TITLE, OVERVIEW, QUALITY, COLUMNS
        # ================================================================
        
        # 1. TITLE
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            alignment=TA_CENTER,
            spaceAfter=10
        )
        story.append(Paragraph("📊 AI Data Analysis Report", title_style))
        story.append(Spacer(1, 5))
        
        # 2. METADATA
        meta_style = ParagraphStyle(
            'MetaStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
        story.append(Paragraph(f"Dataset: {len(df):,} rows, {len(df.columns)} columns", meta_style))
        
        if is_cleaned:
            story.append(Paragraph("<b>✅ Dataset is CLEANED</b>", meta_style))
        else:
            story.append(Paragraph("<i>⚠️ Dataset is ORIGINAL (not cleaned)</i>", meta_style))
        
        story.append(Spacer(1, 20))
        
        # 3. DATASET OVERVIEW
        overview_style = ParagraphStyle(
            'OverviewStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10
        )
        story.append(Paragraph("1. Dataset Overview", overview_style))
        
        summary_data = [
            ["Metric", "Value"],
            ["Total Records", f"{len(df):,}"],
            ["Total Columns", f"{len(df.columns)}"],
            ["Numeric Columns", f"{len(numeric_cols)}"],
            ["Categorical Columns", f"{len(categorical_cols)}"],
            ["Duplicate Rows", f"{df.duplicated().sum():,}"],
            ["Missing Values", f"{missing_cells:,}"],
            ["Data Quality Score", f"{quality_score:.1f}%"],
            ["Cleaning Status", "✅ Cleaned" if is_cleaned else "⚠️ Original"],
        ]
        
        summary_table = Table(summary_data, colWidths=[200, 200])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (1, -1), colors.beige),
            ('GRID', (0, 0), (1, -1), 1, colors.grey),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # 4. DATA QUALITY METRICS
        quality_style = ParagraphStyle(
            'QualityStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10
        )
        story.append(Paragraph("2. Data Quality Metrics", quality_style))
        
        quality_data = [
            ["Metric", "Value"],
            ["Data Quality Score", f"{quality_score:.1f}%"],
            ["Total Cells", f"{total_cells:,}"],
            ["Missing Cells", f"{missing_cells:,}"],
            ["Missing Percentage", f"{(missing_cells / total_cells * 100):.2f}%"],
            ["Duplicate Rows", f"{df.duplicated().sum():,}"],
        ]
        
        quality_table = Table(quality_data, colWidths=[200, 200])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (1, -1), colors.beige),
            ('GRID', (0, 0), (1, -1), 1, colors.grey),
        ]))
        story.append(quality_table)
        story.append(Spacer(1, 20))
        
        # 5. COLUMNS
        col_style = ParagraphStyle(
            'ColStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10
        )
        story.append(Paragraph("3. Columns", col_style))
        
        columns = summary.get('column_names', [])
        col_data = [["#", "Column Name", "Data Type"]]
        for i, col in enumerate(columns, 1):
            dtype = summary.get('data_types', {}).get(col, 'unknown')
            col_data.append([str(i), col, dtype])
        
        col_table = Table(col_data, colWidths=[40, 200, 120])
        col_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (2, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (2, 0), 10),
            ('BOTTOMPADDING', (0, 0), (2, 0), 10),
            ('BACKGROUND', (0, 1), (2, -1), colors.beige),
            ('GRID', (0, 0), (2, -1), 1, colors.grey),
            ('FONTSIZE', (0, 1), (2, -1), 9),
        ]))
        story.append(col_table)
        story.append(Spacer(1, 20))
        
        # ================================================================
        # PAGE 2: NUMERIC STATISTICS, CATEGORICAL STATISTICS, PREVIEW
        # ================================================================
        story.append(PageBreak())
        
        # 6. NUMERIC STATISTICS
        num_style = ParagraphStyle(
            'NumStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10
        )
        story.append(Paragraph("4. Numeric Statistics", num_style))
        
        if summary.get('numeric_stats'):
            num_stats = summary.get('numeric_stats', {})
            num_data = [["Column", "Min", "Max", "Mean", "Median", "Std Dev"]]
            for col, stats in list(num_stats.items())[:10]:
                num_data.append([
                    col,
                    f"{stats.get('min', 0):.2f}",
                    f"{stats.get('max', 0):.2f}",
                    f"{stats.get('mean', 0):.2f}",
                    f"{stats.get('median', 0):.2f}",
                    f"{stats.get('std', 0):.2f}"
                ])
            
            num_table = Table(num_data, colWidths=[100, 70, 70, 70, 70, 70])
            num_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (5, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (5, 0), colors.white),
                ('ALIGN', (0, 0), (5, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (5, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (5, 0), 9),
                ('BOTTOMPADDING', (0, 0), (5, 0), 8),
                ('BACKGROUND', (0, 1), (5, -1), colors.beige),
                ('GRID', (0, 0), (5, -1), 1, colors.grey),
                ('FONTSIZE', (0, 1), (5, -1), 8),
            ]))
            story.append(num_table)
        else:
            story.append(Paragraph("No numeric columns found.", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 7. CATEGORICAL STATISTICS
        cat_style = ParagraphStyle(
            'CatStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10
        )
        story.append(Paragraph("5. Categorical Statistics", cat_style))
        
        if categorical_cols:
            for col in categorical_cols[:3]:
                story.append(Paragraph(f"<b>{col}</b>", styles['Normal']))
                value_counts = df[col].value_counts().head(5)
                cat_data = [["Value", "Count", "Percentage"]]
                for val, count in value_counts.items():
                    pct = (count / len(df)) * 100
                    cat_data.append([str(val), str(count), f"{pct:.1f}%"])
                
                cat_table = Table(cat_data, colWidths=[200, 100, 100])
                cat_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (2, 0), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
                    ('ALIGN', (0, 0), (2, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (2, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (2, 0), 8),
                    ('BACKGROUND', (0, 1), (2, -1), colors.beige),
                    ('GRID', (0, 0), (2, -1), 1, colors.grey),
                    ('FONTSIZE', (0, 1), (2, -1), 9),
                ]))
                story.append(cat_table)
                story.append(Spacer(1, 10))
            
            if len(categorical_cols) > 3:
                story.append(Paragraph(f"<i>... and {len(categorical_cols) - 3} more categorical columns</i>", styles['Normal']))
        else:
            story.append(Paragraph("No categorical columns found.", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 8. DATASET PREVIEW
        preview_style = ParagraphStyle(
            'PreviewStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10
        )
        story.append(Paragraph("6. Dataset Preview (First 5 Rows)", preview_style))
        
        preview_cols = df.columns.tolist()
        if len(preview_cols) > 8:
            preview_cols = preview_cols[:8]
            story.append(Paragraph("<i>Showing first 8 columns only</i>", styles['Normal']))
        
        preview_data = [preview_cols]
        for _, row in df.head(5).iterrows():
            preview_data.append([str(val)[:12] for val in row[preview_cols].values])
        
        col_width = min(70, 500 // len(preview_cols))
        preview_table = Table(preview_data, colWidths=[col_width] * len(preview_cols))
        preview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 6),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 5),
        ]))
        story.append(preview_table)
        story.append(Spacer(1, 20))
        
        # ================================================================
        # PAGE 3: AI INSIGHTS, CHART, EXECUTIVE SUMMARY
        # ================================================================
        story.append(PageBreak())
        
        # 9. AI-GENERATED INSIGHTS
        ai_style = ParagraphStyle(
            'AIStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10
        )
        story.append(Paragraph("7. AI-Generated Insights", ai_style))
        
        try:
            prompt = f"""
            You are a data analysis expert. Analyze the following dataset and provide a comprehensive, easy-to-understand explanation.
            
            DATASET OVERVIEW:
            - Total Rows: {len(df)}
            - Columns: {', '.join(df.columns.tolist())}
            - Numeric Columns: {len(numeric_cols)}
            - Categorical Columns: {len(categorical_cols)}
            
            KEY STATISTICS:
            - Data Quality Score: {quality_score:.1f}%
            - Duplicate Rows: {df.duplicated().sum()}
            - Missing Values: {df.isnull().sum().sum()}
            
            IMPORTANT: Do NOT use Markdown formatting.
            
            Structure your response exactly like this:
            
            DATASET OVERVIEW:
            [Write 2-3 sentences]
            
            KEY FINDINGS:
            - [Finding 1]
            - [Finding 2]
            - [Finding 3]
            - [Finding 4]
            
            NOTABLE PATTERNS:
            - [Pattern 1]
            - [Pattern 2]
            - [Pattern 3]
            
            RECOMMENDATIONS:
            - [Recommendation 1]
            - [Recommendation 2]
            - [Recommendation 3]
            """
            
            llm_response = call_llm(prompt)
            
            if llm_response:
                para_style = ParagraphStyle(
                    'InsightPara',
                    parent=styles['Normal'],
                    fontSize=10,
                    textColor=colors.HexColor('#333333'),
                    alignment=TA_JUSTIFY,
                    spaceAfter=6
                )
                
                section_style = ParagraphStyle(
                    'SectionStyle',
                    parent=styles['Normal'],
                    fontSize=11,
                    textColor=colors.HexColor('#667eea'),
                    fontName='Helvetica-Bold',
                    spaceAfter=4
                )
                
                bullet_style = ParagraphStyle(
                    'BulletStyle',
                    parent=styles['Normal'],
                    fontSize=10,
                    textColor=colors.HexColor('#333333'),
                    alignment=TA_JUSTIFY,
                    spaceAfter=3,
                    leftIndent=20
                )
                
                lines = llm_response.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        story.append(Spacer(1, 4))
                        continue
                    
                    if line.isupper() and len(line) > 10:
                        story.append(Paragraph(f"<b>{line}</b>", section_style))
                    elif line.startswith('-') or line.startswith('•'):
                        clean_line = line.lstrip('-• ').strip()
                        story.append(Paragraph(f"• {clean_line}", bullet_style))
                    else:
                        story.append(Paragraph(line, para_style))
                
                story.append(Spacer(1, 10))
            else:
                story.append(Paragraph("AI insights not available. Please check your DeepSeek API key.", styles['Normal']))
        except Exception as e:
            story.append(Paragraph(f"AI insights not available: {str(e)}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 10. CHART
        chart_style = ParagraphStyle(
            'ChartStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10
        )
        story.append(Paragraph("8. Chart with AI Explanation", chart_style))
        
        if chart_path and os.path.exists(chart_path):
            try:
                from reportlab.platypus import Image
                img = ImageReader(chart_path)
                story.append(Image(chart_path, width=500, height=280))
                story.append(Spacer(1, 10))
                
                chart_exp_style = ParagraphStyle(
                    'ChartExpStyle',
                    parent=styles['Normal'],
                    fontSize=10,
                    textColor=colors.HexColor('#333333'),
                    alignment=TA_JUSTIFY,
                    spaceAfter=6
                )
                
                # FIXED: use the same context builder as /explain-chart so every
                # chart type (Pie, Histogram, Boxplot, Scatter, Line, Heatmap, Bar,
                # Auto) gets real data context, not just Bar/Auto with both axes set.
                chart_context = build_chart_context(df, chart_type, x_column, y_column, numeric_cols, categorical_cols, summary)
                
                chart_prompt = f"""
                You are a data analysis expert. Analyze this chart and provide a clear, informative explanation.
                
                CHART TYPE: {chart_type}
                {chart_context}
                
                Provide a 2-3 sentence explanation that:
                1. Tells what this chart shows
                2. Points out the most important observation
                3. Gives one key insight
                """
                
                chart_exp = call_llm(chart_prompt)
                if chart_exp:
                    story.append(Paragraph(f"<b>Chart Analysis:</b> {chart_exp}", chart_exp_style))
                else:
                    story.append(Paragraph("Chart analysis not available.", chart_exp_style))
                    
            except Exception as e:
                story.append(Paragraph(f"Chart not available: {str(e)}", styles['Normal']))
        else:
            story.append(Paragraph("No chart generated. Please generate a chart first.", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 11. EXECUTIVE SUMMARY
        exec_style = ParagraphStyle(
            'ExecStyle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=10
        )
        story.append(Paragraph("9. Executive Summary", exec_style))
        
        exec_summary = f"""
        <b>Dataset Overview:</b> This dataset contains {len(df):,} records with {len(df.columns)} columns.
        It includes {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.
        The data quality score is {quality_score:.1f}%, with {df.duplicated().sum():,} duplicate rows
        and {df.isnull().sum().sum():,} missing values.
        <br/><br/>
        <b>Key Findings:</b>
        """
        for col in numeric_cols[:5]:
            exec_summary += f"• <b>{col}:</b> Mean={df[col].mean():.2f}, Median={df[col].median():.2f}, Max={df[col].max():.2f}<br/>"
        
        exec_para_style = ParagraphStyle(
            'ExecParaStyle',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        story.append(Paragraph(exec_summary, exec_para_style))
        story.append(Spacer(1, 20))
        
        # 12. FOOTER
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph("Report generated by AI Data Analysis Assistant", footer_style))
        story.append(Paragraph(f"© {datetime.now().year} Saylani Hackathon | All Rights Reserved", footer_style))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary files
        if chart and chart.filename and os.path.exists(chart_path):
            try:
                os.remove(chart_path)
            except:
                pass
        
        return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))
    
    except Exception as e:
        print(f"PDF Generation Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")