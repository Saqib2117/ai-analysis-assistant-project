"""
FastAPI Backend for AI Data Analysis Assistant
With DeepSeek LLM Integration + Chart Explanation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
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

def call_llm(prompt):
    """Call DeepSeek API"""
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
            "max_tokens": 800
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
        if request and request.chart_type:
            chart_type = request.chart_type
        
        df = analyzer.df
        summary = analyzer.get_summary()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # --- BUILD CHART DATA DESCRIPTION ---
        chart_info = ""
        
        if chart_type == "Histogram" or chart_type == "Auto":
            if numeric_cols:
                col = numeric_cols[0]
                stats = summary.get('numeric_stats', {}).get(col, {})
                # Get actual distribution data
                hist_values, bin_edges = np.histogram(df[col].dropna(), bins=10)
                hist_data = ""
                for i in range(len(hist_values)):
                    if i < 5:  # Show top 5 bins
                        hist_data += f"- {bin_edges[i]:.1f} to {bin_edges[i+1]:.1f}: {hist_values[i]} people\n"
                
                chart_info = f"""
                CHART TYPE: Histogram
                COLUMN: {col}
                TOTAL RECORDS: {len(df)}
                
                DISTRIBUTION (Top 5 bins):
                {hist_data}
                
                KEY STATISTICS:
                - Min: {stats.get('min', 'N/A')}
                - Max: {stats.get('max', 'N/A')}
                - Mean: {stats.get('mean', 'N/A'):.2f}
                - Median: {stats.get('median', 'N/A'):.2f}
                - Standard Deviation: {stats.get('std', 'N/A'):.2f}
                """
        
        elif chart_type == "Bar":
            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                top_5 = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(5)
                
                chart_info = f"""
                CHART TYPE: Bar Chart
                X-AXIS: {cat_col}
                Y-AXIS: {num_col}
                
                TOP 5 CATEGORIES:
                """
                for cat, val in top_5.items():
                    chart_info += f"- {cat}: {val:,.0f}\n"
        
        elif chart_type == "Pie":
            if categorical_cols:
                cat_col = categorical_cols[0]
                top_5 = df[cat_col].value_counts().head(5)
                
                chart_info = f"""
                CHART TYPE: Pie Chart
                COLUMN: {cat_col}
                
                TOP 5 CATEGORIES:
                """
                for cat, count in top_5.items():
                    pct = (count / len(df)) * 100
                    chart_info += f"- {cat}: {count} ({pct:.1f}%)\n"
        
        elif chart_type == "Scatter":
            if len(numeric_cols) >= 2:
                col1 = numeric_cols[0]
                col2 = numeric_cols[1]
                chart_info = f"""
                CHART TYPE: Scatter Plot
                X-AXIS: {col1}
                Y-AXIS: {col2}
                TOTAL POINTS: {len(df)}
                
                KEY STATISTICS:
                - {col1} - Min: {df[col1].min():.2f}, Max: {df[col1].max():.2f}, Mean: {df[col1].mean():.2f}
                - {col2} - Min: {df[col2].min():.2f}, Max: {df[col2].max():.2f}, Mean: {df[col2].mean():.2f}
                """
        
        prompt = f"""
        You are a data analysis expert. Analyze this chart and explain what it shows.
        
        {chart_info}
        
        Please provide:
        1. What this chart shows in simple terms (1-2 sentences)
        2. The most prominent observation from this chart (1 sentence)
        3. One key insight from this chart (1 sentence)
        
        Keep it short and focused ONLY on the chart.
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
                "explanation": "⚠️ Failed to generate explanation.",
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
    """Generate AI explanation for the entire dataset"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    if not LLM_AVAILABLE:
        return JSONResponse({
            "success": False,
            "explanation": "⚠️ DeepSeek API key not configured.",
            "llm_available": False
        })
    
    try:
        summary = analyzer.get_summary()
        columns_info = ", ".join(summary.get('column_names', []))
        total_rows = summary.get('total_rows', 0)
        
        # Get numeric stats
        numeric_stats = summary.get('numeric_stats', {})
        stats_text = ""
        for col, stats in numeric_stats.items():
            stats_text += f"- {col}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}, median={stats['median']:.2f}\n"
        
        # Get categorical columns
        categorical_cols = [col for col, dtype in summary.get('data_types', {}).items() if dtype == 'object']
        cat_text = ", ".join(categorical_cols) if categorical_cols else "None"
        
        # Get sample data
        sample_data = analyzer.df.head(5).to_string()
        
        prompt = f"""
        You are a data analysis expert. Analyze the following dataset and provide a comprehensive, easy-to-understand explanation.
        
        DATASET OVERVIEW:
        - Total Rows: {total_rows}
        - Columns: {columns_info}
        - Categorical Columns: {cat_text}
        
        NUMERIC STATISTICS:
        {stats_text}
        
        SAMPLE DATA (First 5 rows):
        {sample_data}
        
        Please provide a complete analysis with:
        
        1. DATASET OVERVIEW (2-3 sentences):
           What kind of data is this? What does it represent?
        
        2. KEY STATISTICS (2-3 bullet points):
           Highlight the most important numbers from the data.
        
        3. NOTABLE PATTERNS (2-3 bullet points):
           What interesting patterns or relationships do you see?
        
        4. SUMMARY (1-2 sentences):
           A simple summary that a non-technical person would understand.
        
        Make it professional but easy to understand.
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
                "explanation": "⚠️ Failed to generate explanation.",
                "llm_available": False
            })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "explanation": f"Error: {str(e)}",
            "llm_available": False
        })

@app.get("/export-pdf")
async def export_pdf():
    """Export analysis as PDF (Bonus Feature)"""
    if not analyzer.is_loaded:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    global chart_path
    if not chart_path:
        chart_gen = ChartGenerator(analyzer.df)
        chart_path = chart_gen.generate_chart()
    
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"analysis_report_{timestamp}.pdf"
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "AI Data Analysis Report")
    
    summary = analyzer.get_summary()
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 95, f"Rows: {summary.get('total_rows', 0)}, Columns: {summary.get('total_columns', 0)}")
    
    y = height - 130
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Dataset Summary")
    y -= 25
    
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Total Records: {summary.get('total_rows', 0)}")
    y -= 15
    c.drawString(50, y, f"Total Columns: {summary.get('total_columns', 0)}")
    y -= 15
    c.drawString(50, y, f"Duplicate Rows: {summary.get('duplicate_rows', 0)}")
    y -= 25
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Columns:")
    y -= 20
    c.setFont("Helvetica", 10)
    for col in summary.get('column_names', [])[:10]:
        c.drawString(50, y, f"  • {col}")
        y -= 15
    
    try:
        img = ImageReader(chart_path)
        c.drawImage(img, 50, y - 250, width=500, height=250, preserveAspectRatio=True)
    except:
        c.drawString(50, y, "Chart not available")
    
    c.save()
    return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_available": LLM_AVAILABLE,
        "llm_provider": LLM_PROVIDER
    }