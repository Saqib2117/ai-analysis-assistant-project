"""
AI Data Analysis Assistant - Premium Dark UI
Deploy on Streamlit Cloud
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re

# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="AI Data Analysis Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS - DARK & MODERN THEME
# ================================================================

st.markdown("""
<style>
    /* ===== GLOBAL THEME ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #12122a 50%, #1a1a3e 100%);
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a3e;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* ===== HEADERS ===== */
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        letter-spacing: -0.5px;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.3);
    }
    .sub-header {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        border: 1px solid rgba(102, 126, 234, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #667eea, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: 2px;
        color: #8892b0;
    }
    
    /* ===== CHAT BUBBLES - FIXED SPACING ===== */
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 5px 18px;
        max-width: 75%;
        margin-left: auto;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.25);
        word-wrap: break-word;
        font-size: 0.95rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .assistant-msg {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        color: #e0e6f0;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 5px;
        max-width: 75%;
        margin-right: auto;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
        word-wrap: break-word;
        font-size: 0.95rem;
    }
    .assistant-msg strong {
        color: #a78bfa;
    }
    .assistant-msg small {
        color: #64748b;
        font-size: 0.8rem;
        display: block;
        margin-top: 0.4rem;
    }
    
    /* ===== TABLES INSIDE CHAT ===== */
    .assistant-msg table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        overflow: hidden;
        font-size: 0.9rem;
    }
    .assistant-msg th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        text-align: left;
    }
    .assistant-msg td {
        padding: 0.4rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        color: #e0e6f0;
    }
    .assistant-msg tr:last-child td {
        border-bottom: none;
    }
    .assistant-msg tr:hover td {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* ===== KEY INSIGHT ===== */
    .key-insight {
        background: rgba(247, 201, 72, 0.12);
        padding: 0.6rem 1rem;
        border-radius: 10px;
        border-left: 4px solid #f7c948;
        margin-top: 0.6rem;
        color: #f7c948;
        font-weight: 500;
        font-size: 0.9rem;
        backdrop-filter: blur(5px);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
        height: 42px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* ===== ALIGNMENT FIXES - ASK & GENERATE BUTTONS ===== */
    div[data-testid="column"] .stButton {
        margin-top: 20px !important;
    }
    
    div[data-testid="column"] .stButton > button {
        margin-top: 0 !important;
    }
    
    /* Fix for columns with buttons */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 4px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 500 !important;
        color: #8892b0 !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.15) !important;
        color: #a78bfa !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* ===== AI BOX ===== */
    .ai-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 5px solid #667eea;
        margin-top: 0.5rem;
        color: white;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .ai-box h4 {
        color: #a78bfa !important;
    }
    .ai-box p {
        color: #e0e6f0 !important;
        line-height: 1.6;
    }
    
    /* ===== SIDEBAR ===== */
    .css-1d391kg, .css-1v3fvcr {
        background: rgba(10, 10, 30, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #e0e6f0 !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: rgba(102, 126, 234, 0.4) !important;
    }
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.03) !important;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 2rem;
        font-size: 0.85rem;
    }
    
    /* ===== SPACING HELPERS ===== */
    .section-spacing {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* ===== INPUT FIELDS ===== */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        background: rgba(255, 255, 255, 0.05) !important;
        color: #e0e6f0 !important;
        height: 42px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #64748b !important;
    }
    
    /* ===== SELECT BOX ===== */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.05) !important;
        color: #e0e6f0 !important;
        height: 42px !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }
    
    /* ===== WELCOME SCREEN ===== */
    .welcome-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        max-width: 600px;
        margin: 0 auto;
        box-shadow: 0 8px 40px rgba(0,0,0,0.3);
    }
    .welcome-box h2 {
        color: #a78bfa;
        margin-top: 1rem;
    }
    .welcome-box p {
        color: #8892b0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .welcome-guide {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 14px;
        margin-top: 1.5rem;
        text-align: left;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .welcome-guide p {
        color: #e0e6f0 !important;
        font-size: 0.95rem !important;
    }
    .welcome-guide strong {
        color: #a78bfa;
    }
    
    .stSubheader {
        color: #e0e6f0 !important;
    }
    
    code {
        color: #a78bfa !important;
        background: rgba(255, 255, 255, 0.05) !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
    }
    
    /* ===== CHAT DIVIDER SPACING ===== */
    .chat-divider {
        margin: 20px 0 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SESSION STATE
# ================================================================

BASE_URL = "https://saqib21-fastapi-backend.hf.space"

if "api_url" not in st.session_state:
    st.session_state.api_url = BASE_URL
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'chart_path' not in st.session_state:
    st.session_state.chart_path = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'llm_available' not in st.session_state:
    st.session_state.llm_available = False
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "None"
if 'question_input' not in st.session_state:
    st.session_state.question_input = ""
if '_temp_question' not in st.session_state:
    st.session_state._temp_question = ""
if 'last_chart_type' not in st.session_state:
    st.session_state.last_chart_type = "Auto"

# ================================================================
# HELPER FUNCTIONS - FIXED TABLE FORMATTING
# ================================================================

def clean_markdown(text):
    """Remove ALL ** markers and clean text for display"""
    if not text:
        return ""
    text = text.replace('**', '')
    text = ' '.join(text.split())
    return text

def update_summary_after_cleaning(df):
    """Update the summary after cleaning"""
    try:
        # Calculate new summary
        new_summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
        }
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            new_summary["numeric_stats"] = {}
            for col in numeric_cols:
                new_summary["numeric_stats"][col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                }
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            new_summary["categorical_stats"] = {}
            for col in categorical_cols:
                new_summary["categorical_stats"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(5).to_dict(),
                }
        
        # Update session state
        st.session_state.summary = new_summary
        
        return True
    except Exception as e:
        print(f"Error updating summary: {e}")
        return False

def format_answer_display(answer_text, explanation=""):
    """Format answer with proper styling - FIXED TABLE HANDLING"""
    
    clean_explanation = clean_markdown(explanation) if explanation else ""
    
    # Check if it's a table (contains | or has Category/Value pattern)
    if "|" in answer_text or ("Category" in answer_text and "Value" in answer_text):
        lines = answer_text.split('\n')
        data_lines = []
        insight = ""
        header = ""
        is_table = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Clean the header
            if line.startswith("📊"):
                header = line.replace("📊", "").strip()
                header = clean_markdown(header)
                is_table = True
            elif "|" in line and "Category" in line:
                is_table = True
                continue
            elif "|" in line and "---" in line:
                continue
            elif "|" in line:
                # Data row with | separator
                is_table = True
                clean_line = clean_markdown(line)
                clean_line = clean_line.replace('|', '').strip()
                if clean_line:
                    data_lines.append(clean_line)
            elif "💡" in line:
                insight = line.replace("💡", "").strip()
                insight = clean_markdown(insight)
                insight = insight.replace("|", " | ")
            elif line and "Category" not in line and "Value" not in line:
                # Check if it's a data row without | (e.g., "Some-college 35.68")
                parts = line.split()
                if len(parts) >= 2:
                    # Check if last part is a number (value)
                    try:
                        float(parts[-1])
                        # It's a data row: category + value
                        is_table = True
                        data_lines.append(line)
                    except ValueError:
                        # Not a data row, skip
                        pass
        
        formatted = ""
        if header:
            formatted += f"<strong>{header}</strong><br><br>"
        
        if data_lines and is_table:
            formatted += "<table>"
            formatted += "<tr><th>Category</th><th>Value</th></tr>"
            for row in data_lines:
                # Split by space to get category and value
                parts = row.split()
                if len(parts) >= 2:
                    # Check if last part is a number
                    try:
                        value = parts[-1]
                        # The category is everything except the last part
                        category = " ".join(parts[:-1])
                        formatted += f"<tr><td>{category}</td><td>{value}</td></tr>"
                    except ValueError:
                        # If not a number, treat the whole line as category
                        formatted += f"<tr><td>{row}</td><td></td></tr>"
            formatted += "</table>"
        
        if insight:
            formatted += f"<div class='key-insight'>💡 {insight}</div>"
        
        if clean_explanation:
            formatted += f"<small>{clean_explanation}</small>"
        
        return formatted
    
    else:
        # Simple answer - clean completely
        clean_answer = clean_markdown(answer_text)
        if clean_explanation:
            return f"{clean_answer}<br><small>{clean_explanation}</small>"
        return clean_answer

# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 3.5rem;">📊</div>
        <h2 style="color: #a78bfa; margin: 0.2rem 0 0 0; font-size: 1.5rem;">AI Assistant</h2>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0;">Data Analysis Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    llm_status = "✅ Connected" if st.session_state.llm_available else "⚠️ Not Configured"
    llm_color = "#34d399" if st.session_state.llm_available else "#f59e0b"
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 0.7rem; border-radius: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.06);">
        <span style="font-size: 1.2rem;">🤖</span>
        <span style="font-weight: 600; color: #a78bfa;">DeepSeek AI</span>
        <br>
        <span style="font-size: 0.75rem; color: {llm_color};">{llm_status}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <h4 style="color: #a78bfa; margin-bottom: 0.5rem; font-size: 1rem;">📤 Upload Dataset</h4>
    <p style="color: #64748b; font-size: 0.8rem;">Upload a CSV file to analyze</p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.info(f"📄 {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
        
        if st.button("🚀 Upload & Analyze", use_column_width=True):
            with st.spinner("Analyzing your data..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                    response = requests.post(
                        f"{st.session_state.api_url}/upload",
                        files=files,
                        timeout=30
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.data_loaded = True
                        st.session_state.summary = data.get('summary')
                        st.session_state.llm_available = data.get('llm_available', False)
                        st.session_state.llm_provider = data.get('llm_provider', 'None')
                        
                        chart_resp = requests.get(f"{st.session_state.api_url}/chart")
                        if chart_resp.status_code == 200:
                            os.makedirs("temp", exist_ok=True)
                            chart_path = "temp/chart.png"
                            with open(chart_path, "wb") as f:
                                f.write(chart_resp.content)
                            st.session_state.chart_path = chart_path
                        
                        st.success("✅ Dataset loaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    st.markdown("""
    <h4 style="color: #a78bfa; margin-bottom: 0.5rem; font-size: 1rem;">✨ Features</h4>
    """, unsafe_allow_html=True)
    
    features = ["📊 Auto Charts", "❓ Natural Language Q&A", "🤖 AI Explanations", "📄 PDF Export", "📥 Chart Download"]
    for f in features:
        st.markdown(f"<p style='font-size: 0.85rem; color: #8892b0; margin: 0.2rem 0;'>• {f}</p>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown(f"""
    <p style="font-size: 0.75rem; color: #64748b; text-align: center;">
        🔗 Backend: Hugging Face<br>
        ⚡ Status: <span style="color: #34d399;">● Online</span>
    </p>
    """, unsafe_allow_html=True)

# ================================================================
# MAIN CONTENT
# ================================================================

st.markdown('<h1 class="main-header">📊 AI Data Analysis Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your CSV and ask questions in natural language</p>', unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.markdown("""
    <div class="welcome-box">
        <div style="font-size: 4rem;">🚀</div>
        <h2>Welcome to AI Data Analysis</h2>
        <p>Upload your CSV file to get started with AI-powered analysis</p>
        <div class="welcome-guide">
            <p>📌 <strong>Quick Start Guide</strong></p>
            <p style="font-size: 0.9rem !important;">
                1️⃣ Upload a CSV file from the sidebar<br>
                2️⃣ View automatic dataset summary<br>
                3️⃣ Ask questions about your data<br>
                4️⃣ Generate professional charts<br>
                5️⃣ Export PDF reports
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    summary = st.session_state.summary
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('total_rows', 0):,}</div>
            <div class="metric-label">📋 Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{summary.get('total_columns', 0)}</div>
            <div class="metric-label">📑 Total Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing = sum(summary.get('missing_values', {}).values())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{missing:,}</div>
            <div class="metric-label">⚠️ Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        duplicates = summary.get('duplicate_rows', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{duplicates:,}</div>
            <div class="metric-label">🔄 Duplicates</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
    
    # ===== TAB ORDER: Dashboard → Preview Dataset → Data Cleaning → Statistics → Q&A → Charts → AI Explain → Export =====
    tab1, tab6, tab7, tab8, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🔍 Preview Dataset",
        "🧹 Data Cleaning",
        "📊 Statistics & Insights", 
        "💬 Q&A",
        "📈 Charts",
        "🤖 AI Explain",
        "📄 Export"
    ])
    
    # ================================================================
    # TAB 1: Dashboard
    # ================================================================
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Columns & Data Types")
            for col, dtype in summary.get('data_types', {}).items():
                st.markdown(f"<p>• <code>{col}</code>: {dtype}</p>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔢 Numeric Statistics")
            for col, stats in summary.get('numeric_stats', {}).items():
                with st.expander(f"📊 {col}"):
                    st.write(f"Min: {stats['min']:.2f}")
                    st.write(f"Max: {stats['max']:.2f}")
                    st.write(f"Mean: {stats['mean']:.2f}")
                    st.write(f"Median: {stats['median']:.2f}")
                    st.write(f"Std: {stats['std']:.2f}")
        
        st.divider()
        st.subheader("📈 Chart Preview")
        if st.session_state.chart_path:
            st.image(st.session_state.chart_path, use_column_width=True)
        else:
            if st.button("🔄 Generate Chart"):
                try:
                    response = requests.get(f"{st.session_state.api_url}/chart")
                    if response.status_code == 200:
                        os.makedirs("temp", exist_ok=True)
                        chart_path = "temp/chart.png"
                        with open(chart_path, "wb") as f:
                            f.write(response.content)
                        st.session_state.chart_path = chart_path
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ================================================================
    # TAB 6: Preview Dataset
    # ================================================================

    with tab6:
        st.subheader("📊 Dataset Preview")
        
        if st.session_state.data_loaded:
            try:
                # Get dataset info
                response = requests.get(f"{st.session_state.api_url}/info")
                if response.status_code == 200:
                    data = response.json()
                    info = data.get('info', {})
                    
                    # Get summary for duplicates and missing values
                    summary = st.session_state.summary
                    
                    # Display basic info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📋 Total Rows", info.get('shape', [0, 0])[0])
                    with col2:
                        st.metric("📑 Total Columns", info.get('shape', [0, 0])[1])
                    with col3:
                        duplicates = summary.get('duplicate_rows', 0) if summary else 0
                        st.metric("🔄 Duplicate Rows", f"{duplicates:,}")
                    with col4:
                        missing = sum(summary.get('missing_values', {}).values()) if summary else 0
                        st.metric("⚠️ Missing Values", f"{missing:,}")
                    
                    st.divider()
                    
                    # --- SEARCH COLUMNS WITH DATA PREVIEW ---
                    st.subheader("🔍 Search Columns")
                    col_search = st.text_input("Search column names", placeholder="Type column name to search...")
                    
                    # Display columns
                    columns = info.get('columns', [])
                    
                    # --- GET FULL DATASET FOR PREVIEW ---
                    try:
                        response_full = requests.get(f"{st.session_state.api_url}/full-dataset")
                        if response_full.status_code == 200:
                            full_data = response_full.json()
                            df_full = pd.DataFrame(full_data.get('data', []))
                            
                            if not df_full.empty:
                                
                                # ================================================================
                                # IF SEARCH IS ACTIVE - SHOW COLUMN DATA PREVIEW
                                # ================================================================
                                if col_search:
                                    filtered_cols = [col for col in columns if col_search.lower() in col.lower()]
                                    
                                    if filtered_cols:
                                        st.success(f"✅ Found {len(filtered_cols)} column(s) matching '{col_search}'")
                                        
                                        for col in filtered_cols:
                                            with st.expander(f"📊 Column: `{col}`", expanded=True):
                                                # Get column data
                                                col_data = df_full[col]
                                                col_type = col_data.dtype
                                                unique_count = col_data.nunique()
                                                null_count = col_data.isnull().sum()
                                                
                                                # Column stats
                                                c1, c2, c3 = st.columns(3)
                                                with c1:
                                                    st.metric("📋 Data Type", str(col_type))
                                                with c2:
                                                    st.metric("🔢 Unique Values", unique_count)
                                                with c3:
                                                    st.metric("⚠️ Null Values", null_count)
                                                
                                                # --- FIX: Show first 10 values with Index first ---
                                                st.write("**First 10 values:**")
                                                
                                                # Create a clean display of first 10 values
                                                first_10 = col_data.head(10).reset_index(drop=True)
                                                
                                                # Display as a simple table with Index column first
                                                display_df = pd.DataFrame({
                                                    'Index': first_10.index,
                                                    'Value': first_10.values
                                                })
                                                # Reorder columns to put Index first
                                                display_df = display_df[['Index', 'Value']]
                                                st.dataframe(display_df, use_column_width=True, hide_index=True)
                                                
                                                # --- Show value counts for categorical columns ---
                                                if col_type == 'object' or col_type.name == 'category' or col_type == 'str':
                                                    st.write("**Top 5 Most Frequent Values:**")
                                                    value_counts = col_data.value_counts().head(5)
                                                    for val, count in value_counts.items():
                                                        pct = (count / len(col_data)) * 100
                                                        st.write(f"• `{val}`: {count:,} ({pct:.1f}%)")
                                                
                                                st.divider()
                                    else:
                                        st.warning(f"❌ No columns found matching '{col_search}'")
                                        
                                        # Show similar columns suggestion
                                        suggestions = [col for col in columns if any(word in col.lower() for word in col_search.lower().split())]
                                        if suggestions:
                                            st.write("**Did you mean?**")
                                            for col in suggestions[:5]:
                                                st.markdown(f"• `{col}`")
                                
                                # ================================================================
                                # IF NO SEARCH - SHOW FULL PREVIEW
                                # ================================================================
                                else:
                                    # Show column list
                                    st.write(f"**{len(columns)} columns:**")
                                    cols_per_row = 4
                                    for i in range(0, len(columns), cols_per_row):
                                        row_cols = st.columns(cols_per_row)
                                        for j, col in enumerate(columns[i:i+cols_per_row]):
                                            with row_cols[j]:
                                                st.markdown(f"• `{col}`")
                                    
                                    st.divider()
                                    
                                    # First 5 rows
                                    st.subheader("📄 First 5 Rows")
                                    st.dataframe(df_full.head(5), use_column_width=True)
                                    
                                    st.divider()
                                    
                                    # Last 5 rows
                                    st.subheader("📄 Last 5 Rows")
                                    st.dataframe(df_full.tail(5), use_column_width=True)
                            else:
                                st.info("No data available to preview")
                        else:
                            st.error("Could not fetch full dataset for preview")
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
                        
                else:
                    st.error("Could not fetch dataset info")
                    
            except Exception as e:
                st.error(f"Error loading dataset preview: {str(e)}")
        else:
            st.info("Please upload a dataset first to preview.")
    
    # ================================================================
    # TAB 2: Q&A
    # ================================================================

    with tab2:
        st.subheader("💬 Ask Questions About Your Data")
        
        for item in st.session_state.chat_history:
            if item['type'] == 'question':
                st.markdown(f"""
                <div class="user-msg">
                    <strong>❓ You:</strong><br>{item['text']}
                </div>
                """, unsafe_allow_html=True)
            else:
                answer_text = item['text']
                explanation = item.get('explanation', '')
                formatted_display = format_answer_display(answer_text, explanation)
                
                st.markdown(f"""
                <div class="assistant-msg">
                    <strong>🤖 AI:</strong><br>{formatted_display}
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            question = st.text_input(
                "Ask a question",
                placeholder="e.g., Which product generated the highest sales?",
                key="question_input",
                label_visibility="collapsed"
            )
        
        with col2:
            ask_button = st.button("❓ Ask", use_column_width=True, key="ask_btn")
        
        if ask_button and question:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{st.session_state.api_url}/ask",
                        json={"question": question}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.chat_history.append({
                            'type': 'question',
                            'text': question
                        })
                        st.session_state.chat_history.append({
                            'type': 'answer',
                            'text': result.get('answer', 'No answer found'),
                            'explanation': result.get('explanation', '')
                        })
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        st.markdown("---")
        st.caption("💡 Try these example questions:")
        cols = st.columns(3)
        examples = [
            "Which product has highest sales?",
            "What is the average age?",
            "What is the most frequent category?",
        ]
        
        for i, q in enumerate(examples):
            with cols[i]:
                if st.button(q, key=f"q_{i}", use_column_width=True):
                    st.session_state._temp_question = q
                    st.rerun()
        
        if st.session_state._temp_question:
            question_text = st.session_state._temp_question
            st.session_state._temp_question = ""
            
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(
                        f"{st.session_state.api_url}/ask",
                        json={"question": question_text}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.chat_history.append({
                            'type': 'question',
                            'text': question_text
                        })
                        st.session_state.chat_history.append({
                            'type': 'answer',
                            'text': result.get('answer', 'No answer found'),
                            'explanation': result.get('explanation', '')
                        })
                        st.rerun()
                except Exception as e:
                    st.session_state._temp_question = ""
                    st.error(f"Error: {str(e)}")
    
    # ================================================================
    # TAB 3: Charts
    # ================================================================
    
    with tab3:
        st.subheader("📈 Chart Generation")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Auto", "Bar", "Pie", "Histogram", "Scatter"],
                label_visibility="collapsed"
            )
        
        with col2:
            generate_btn = st.button("🎨 Generate", use_column_width=True, key="generate_chart_btn")
        
        if generate_btn:
            with st.spinner("Generating chart..."):
                try:
                    if chart_type == "Auto":
                        endpoint = f"{st.session_state.api_url}/chart"
                    else:
                        endpoint = f"{st.session_state.api_url}/chart/{chart_type.lower()}"
                    
                    response = requests.get(endpoint)
                    if response.status_code == 200:
                        os.makedirs("temp", exist_ok=True)
                        chart_path = f"temp/{chart_type.lower()}_chart.png"
                        with open(chart_path, "wb") as f:
                            f.write(response.content)
                        st.session_state.chart_path = chart_path
                        st.session_state.last_chart_type = chart_type
                        st.image(chart_path, use_column_width=True)
                        st.success("✅ Chart generated!")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.session_state.chart_path:
            st.image(st.session_state.chart_path, use_column_width=True)
            
            with open(st.session_state.chart_path, "rb") as f:
                st.download_button(
                    label="📥 Download Chart (PNG)",
                    data=f,
                    file_name="chart.png",
                    mime="image/png",
                    use_column_width=True
                )
        
        st.divider()
        st.subheader("🤖 Explain This Chart")
        
        if st.button("📊 Explain This Chart with AI", use_column_width=True):
            with st.spinner("🤖 AI is analyzing the chart..."):
                try:
                    chart_type = st.session_state.get('last_chart_type', 'Auto')
                    
                    response = requests.post(
                        f"{st.session_state.api_url}/explain-chart",
                        json={"chart_type": chart_type}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            st.markdown(f"""
                            <div class="ai-box">
                                <h4>📊 Chart Analysis</h4>
                                <p style="font-size: 1.05rem; line-height: 1.8;">{result.get('explanation', 'No explanation generated')}</p>
                                <p style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">
                                    Powered by {result.get('llm_provider', 'AI')}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("💡 " + result.get('explanation', 'AI explanation is not available.'))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ================================================================
    # TAB 4: AI Explain
    # ================================================================
    
    with tab4:
        st.subheader("📊 Dataset Overview")
        st.markdown("""
        <p style="color: #64748b;">
            Get a comprehensive AI-powered analysis of your entire dataset.
        </p>
        """, unsafe_allow_html=True)
        
        if st.button("📊 Explain My Data with AI", use_column_width=True):
            with st.spinner("🤖 AI is analyzing your dataset..."):
                try:
                    response = requests.post(f"{st.session_state.api_url}/explain-data")
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            st.markdown(f"""
                            <div class="ai-box">
                                <h4>📊 Dataset Analysis</h4>
                                <p style="font-size: 1.05rem; line-height: 1.8;">{result.get('explanation', 'No explanation generated')}</p>
                                <p style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">
                                    Powered by {result.get('llm_provider', 'AI')}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("💡 " + result.get('explanation', 'AI explanation is not available.'))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ================================================================
    # TAB 5: Export
    # ================================================================
    
    with tab5:
        st.subheader("📄 Export Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export PDF Report", use_column_width=True):
                with st.spinner("Generating PDF..."):
                    try:
                        response = requests.get(f"{st.session_state.api_url}/export-pdf")
                        if response.status_code == 200:
                            os.makedirs("exports", exist_ok=True)
                            filename = f"exports/analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            with open(filename, "wb") as f:
                                f.write(response.content)
                            
                            st.success("✅ PDF generated!")
                            with open(filename, "rb") as f:
                                st.download_button(
                                    label="📥 Download PDF",
                                    data=f,
                                    file_name="analysis_report.pdf",
                                    mime="application/pdf",
                                    use_column_width=True
                                )
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("📋 Export Summary (JSON)", use_column_width=True):
                st.json(st.session_state.summary)
    
    # ================================================================
    # TAB 7: Data Cleaning & Info
    # ================================================================

    with tab7:
        st.subheader("🧹 Data Cleaning & Info")
        
        if not st.session_state.data_loaded:
            st.info("Please upload a dataset first to access data cleaning features.")
        else:
            # Initialize cleaning session state
            if 'cleaned_df' not in st.session_state:
                st.session_state.cleaned_df = None
            if 'cleaning_applied' not in st.session_state:
                st.session_state.cleaning_applied = False
            if 'cleaning_message' not in st.session_state:
                st.session_state.cleaning_message = ""
            
            # Get current dataframe (original or cleaned)
            try:
                # Always fetch from backend for original data
                response = requests.get(f"{st.session_state.api_url}/full-dataset")
                if response.status_code == 200:
                    full_data = response.json()
                    original_df = pd.DataFrame(full_data.get('data', []))
                    
                    # Use cleaned data if available, otherwise use original
                    if st.session_state.cleaned_df is not None:
                        df = st.session_state.cleaned_df
                        is_cleaned = True
                    else:
                        df = original_df
                        is_cleaned = False
                    
                    # ============================================================
                    # SECTION 1: DATA INFORMATION
                    # ============================================================
                    st.subheader("📊 Data Information")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Get data info
                    rows, cols = df.shape
                    duplicates = df.duplicated().sum()
                    missing = df.isnull().sum().sum()
                    memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                    
                    # Detect numeric and categorical columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    with col1:
                        st.metric("📋 Rows", f"{rows:,}")
                    with col2:
                        st.metric("📑 Columns", cols)
                    with col3:
                        st.metric("🔄 Duplicates", f"{duplicates:,}")
                    with col4:
                        st.metric("⚠️ Missing Values", f"{missing:,}")
                    
                    # Additional info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔢 Numeric Columns", len(numeric_cols))
                    with col2:
                        st.metric("📝 Categorical Columns", len(categorical_cols))
                    with col3:
                        st.metric("💾 Memory Usage", f"{memory:.2f} MB")
                    
                    # Show cleaned status badge
                    if is_cleaned:
                        st.success("✅ **Currently viewing CLEANED data**")
                    else:
                        st.info("ℹ️ **Viewing ORIGINAL data**")
                    
                    st.divider()
                    
                    # ============================================================
                    # SECTION 2: MISSING VALUES DETAIL
                    # ============================================================
                    st.subheader("🔍 Missing Values by Column")
                    
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0]
                    
                    if len(missing_data) > 0:
                        missing_df = pd.DataFrame({
                            'Column': missing_data.index,
                            'Missing Values': missing_data.values,
                            'Percentage': (missing_data.values / len(df) * 100).round(2)
                        })
                        st.dataframe(missing_df, use_column_width=True)
                    else:
                        st.success("✅ No missing values found in the dataset!")
                    
                    st.divider()
                    
                    # ============================================================
                    # SECTION 3: DATA QUALITY DETECTION
                    # ============================================================
                    st.subheader("🔎 Data Quality Detection")
                    
                    quality_issues = []
                    
                    # Check for duplicates
                    if duplicates > 0:
                        quality_issues.append(f"🔄 {duplicates:,} duplicate rows found")
                    
                    # Check for missing values
                    if missing > 0:
                        quality_issues.append(f"⚠️ {missing:,} missing values found")
                    
                    # Check for empty strings
                    empty_count = 0
                    for col in df.select_dtypes(include=['object']).columns:
                        empty_count += (df[col] == '').sum()
                    if empty_count > 0:
                        quality_issues.append(f"📭 {empty_count:,} empty strings found")
                    
                    # Check for whitespace issues
                    whitespace_count = 0
                    for col in df.select_dtypes(include=['object']).columns:
                        whitespace_count += df[col].astype(str).str.strip().eq('').sum()
                    if whitespace_count > 0:
                        quality_issues.append(f"🔤 {whitespace_count:,} whitespace issues found")
                    
                    if quality_issues:
                        for issue in quality_issues:
                            st.warning(issue)
                    else:
                        st.success("✅ Data quality looks good! No issues detected.")
                    
                    st.divider()
                    
                    # ============================================================
                    # SECTION 4: CLEANING OPTIONS
                    # ============================================================
                    st.subheader("🛠️ Cleaning Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        remove_duplicates = st.checkbox("🗑️ Remove duplicate records")
                        remove_empty = st.checkbox("📭 Remove empty records")
                        trim_whitespace = st.checkbox("✂️ Trim whitespace from text columns")
                    
                    with col2:
                        fill_missing = st.checkbox("📊 Fill missing values")
                        if fill_missing:
                            fill_strategy = st.selectbox(
                                "Fill strategy:",
                                ["Auto", "Mean", "Median", "Mode", "Forward Fill", "Backward Fill", "Custom Value"]
                            )
                            if fill_strategy == "Custom Value":
                                custom_value = st.text_input("Enter custom value:", "0")
                        else:
                            fill_strategy = "None"
                        
                        detect_dates = st.checkbox("📅 Convert date columns")
                    
                    # Apply cleaning button
                    st.divider()
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col2:
                        apply_cleaning = st.button("✅ Apply Cleaning", use_column_width=True, type="primary")
                        reset_data = st.button("🔄 Reset to Original", use_column_width=True)
                    
                    # ============================================================
                    # SECTION 5: APPLY CLEANING
                    # ============================================================
                    if apply_cleaning:
                        try:
                            cleaned_df = df.copy()
                            cleaning_steps = []
                            
                            # 1. Remove duplicates
                            if remove_duplicates:
                                before = len(cleaned_df)
                                cleaned_df = cleaned_df.drop_duplicates()
                                after = len(cleaned_df)
                                cleaning_steps.append(f"Removed {before - after} duplicate rows")
                            
                            # 2. Remove empty records
                            if remove_empty:
                                before = len(cleaned_df)
                                cleaned_df = cleaned_df.dropna(how='all')
                                after = len(cleaned_df)
                                cleaning_steps.append(f"Removed {before - after} empty rows")
                            
                            # 3. Trim whitespace
                            if trim_whitespace:
                                for col in cleaned_df.select_dtypes(include=['object']).columns:
                                    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                                cleaning_steps.append("Trimmed whitespace from text columns")
                            
                            # 4. Fill missing values
                            if fill_missing:
                                if fill_strategy == "Auto":
                                    for col in cleaned_df.columns:
                                        if cleaned_df[col].isnull().sum() > 0:
                                            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                                                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                                            else:
                                                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown")
                                    cleaning_steps.append("Filled missing values using Auto strategy")
                                elif fill_strategy == "Mean":
                                    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                                    cleaning_steps.append("Filled missing values using Mean strategy")
                                elif fill_strategy == "Median":
                                    for col in cleaned_df.select_dtypes(include=[np.number]).columns:
                                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                                    cleaning_steps.append("Filled missing values using Median strategy")
                                elif fill_strategy == "Mode":
                                    for col in cleaned_df.columns:
                                        if cleaned_df[col].isnull().sum() > 0:
                                            mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown"
                                            cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                                    cleaning_steps.append("Filled missing values using Mode strategy")
                                elif fill_strategy == "Forward Fill":
                                    cleaned_df = cleaned_df.fillna(method='ffill')
                                    cleaning_steps.append("Filled missing values using Forward Fill strategy")
                                elif fill_strategy == "Backward Fill":
                                    cleaned_df = cleaned_df.fillna(method='bfill')
                                    cleaning_steps.append("Filled missing values using Backward Fill strategy")
                                elif fill_strategy == "Custom Value":
                                    cleaned_df = cleaned_df.fillna(custom_value)
                                    cleaning_steps.append(f"Filled missing values with custom value: {custom_value}")
                            
                            # 5. Convert date columns
                            if detect_dates:
                                date_cols_detected = []
                                for col in cleaned_df.columns:
                                    if cleaned_df[col].dtype == 'object':
                                        try:
                                            pd.to_datetime(cleaned_df[col])
                                            cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                                            date_cols_detected.append(col)
                                        except:
                                            pass
                                if date_cols_detected:
                                    cleaning_steps.append(f"Converted {len(date_cols_detected)} columns to date: {', '.join(date_cols_detected)}")
                                else:
                                    cleaning_steps.append("No date columns detected")
                            
                            # Store cleaned data
                            st.session_state.cleaned_df = cleaned_df
                            st.session_state.cleaning_applied = True
                            st.session_state.cleaning_message = f"✅ Data cleaned successfully! {len(cleaning_steps)} operations applied."
                            
                            # Update summary after cleaning
                            update_summary_after_cleaning(cleaned_df)
                            
                            # Show success message (ONLY ONE)
                            st.success(st.session_state.cleaning_message)
                            for step in cleaning_steps:
                                st.write(f"• {step}")
                            
                            # Rerun to refresh the tab
                            st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Error during cleaning: {str(e)}")
                    
                    # ============================================================
                    # SECTION 6: RESET DATA
                    # ============================================================
                    if reset_data:
                        st.session_state.cleaned_df = None
                        st.session_state.cleaning_applied = False
                        st.session_state.cleaning_message = "🔄 Dataset reset to original"
                        
                        # Reset summary to original
                        try:
                            response_summary = requests.get(f"{st.session_state.api_url}/summary")
                            if response_summary.status_code == 200:
                                data = response_summary.json()
                                st.session_state.summary = data.get('summary')
                        except:
                            pass
                        
                        st.success("🔄 Dataset reset to original!")
                        st.rerun()
                    
                    # ============================================================
                    # SECTION 7: CLEANING STATUS
                    # ============================================================
                    st.divider()
                    st.subheader("📋 Cleaning Status")
                    
                    if st.session_state.cleaning_applied:
                        # Show cleaned vs original comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Rows", f"{len(original_df):,}")
                        with col2:
                            st.metric("Cleaned Rows", f"{len(st.session_state.cleaned_df):,}")
                        
                        # Show difference
                        diff = len(original_df) - len(st.session_state.cleaned_df)
                        if diff > 0:
                            st.success(f"✅ Removed {diff} rows through cleaning")
                        else:
                            st.info("ℹ️ No rows removed during cleaning")
                    else:
                        st.info("ℹ️ No cleaning applied yet. Use the options above to clean your data.")
                        
                else:
                    st.error("Could not load dataset for cleaning")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # ================================================================
    # TAB 8: Statistics & Insights
    # ================================================================

    with tab8:
        st.subheader("📊 Statistics & Insights")
        
        if not st.session_state.data_loaded:
            st.info("Please upload a dataset first to view statistics.")
        else:
            # Get current dataframe
            response = requests.get(f"{st.session_state.api_url}/full-dataset")
            if response.status_code == 200:
                full_data = response.json()
                df = pd.DataFrame(full_data.get('data', []))
                
                # Use cleaned data if available
                if st.session_state.cleaned_df is not None:
                    df = st.session_state.cleaned_df
                
                # ============================================================
                # SECTION 1: SUMMARY STATISTICS
                # ============================================================
                st.subheader("📋 Summary Statistics")
                
                with st.expander("📊 View Full Summary Statistics", expanded=True):
                    describe_df = df.describe(include='all').T
                    for col in df.select_dtypes(include=['object']).columns:
                        if col in describe_df.index:
                            describe_df.loc[col, 'unique'] = df[col].nunique()
                            describe_df.loc[col, 'top'] = df[col].mode()[0] if not df[col].mode().empty else ''
                            describe_df.loc[col, 'freq'] = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                    st.dataframe(describe_df, use_column_width=True)
                
                st.divider()
                
                # ============================================================
                # SECTION 2: NUMERIC COLUMNS STATISTICS
                # ============================================================
                st.subheader("🔢 Numeric Columns Statistics")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    numeric_stats = []
                    for col in numeric_cols:
                        stats = {
                            'Column': col,
                            'Count': df[col].count(),
                            'Missing': df[col].isnull().sum(),
                            'Mean': df[col].mean(),
                            'Median': df[col].median(),
                            'Std': df[col].std(),
                            'Min': df[col].min(),
                            'Max': df[col].max(),
                            'Q1': df[col].quantile(0.25),
                            'Q3': df[col].quantile(0.75),
                            'Skew': df[col].skew(),
                            'Kurtosis': df[col].kurtosis()
                        }
                        numeric_stats.append(stats)
                    stats_df = pd.DataFrame(numeric_stats)
                    st.dataframe(stats_df.round(2), use_column_width=True)
                else:
                    st.info("No numeric columns found in the dataset.")
                
                st.divider()
                
                # ============================================================
                # SECTION 3: CATEGORICAL COLUMNS STATISTICS
                # ============================================================
                st.subheader("📝 Categorical Columns Statistics")
                
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if categorical_cols:
                    for col in categorical_cols[:5]:
                        with st.expander(f"📊 {col}", expanded=False):
                            value_counts = df[col].value_counts()
                            total = len(df)
                            cat_df = pd.DataFrame({
                                'Value': value_counts.index,
                                'Count': value_counts.values,
                                'Percentage': (value_counts.values / total * 100).round(2)
                            })
                            st.dataframe(cat_df.head(10), use_column_width=True)
                            st.write(f"**Total categories:** {len(value_counts)}")
                            st.write(f"**Most common:** '{value_counts.index[0]}' ({value_counts.iloc[0]} occurrences, {(value_counts.iloc[0]/total*100):.1f}%)")
                    
                    if len(categorical_cols) > 5:
                        st.info(f"Showing 5 of {len(categorical_cols)} categorical columns.")
                else:
                    st.info("No categorical columns found in the dataset.")
                
                st.divider()
                
                # ============================================================
                # SECTION 4: OUTLIER DETECTION (IQR Method)
                # ============================================================
                st.subheader("📊 Outlier Detection (IQR Method)")

                if numeric_cols:
                    outlier_results = []
                    outlier_data = {}
                    
                    for col in numeric_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                        
                        outlier_results.append({
                            'Column': col,
                            'Q1': Q1,
                            'Q3': Q3,
                            'IQR': IQR,
                            'Lower Bound': lower_bound,
                            'Upper Bound': upper_bound,
                            'Outliers Count': len(outliers),
                            'Outlier %': round((len(outliers) / len(df) * 100), 2)
                        })
                        
                        if len(outliers) > 0:
                            outlier_data[col] = {
                                'values': outliers[col].tolist()[:20],
                                'count': len(outliers),
                                'lower_bound': lower_bound,
                                'upper_bound': upper_bound
                            }
                    
                    outlier_df = pd.DataFrame(outlier_results)
                    st.dataframe(outlier_df.round(2), use_column_width=True)
                    
                    columns_with_outliers = outlier_df[outlier_df['Outliers Count'] > 0]
                    
                    if len(columns_with_outliers) > 0:
                        st.warning(f"⚠️ {len(columns_with_outliers)} columns have outliers detected.")
                        st.subheader("🔍 Outlier Details Showcase")
                        
                        for idx, row in columns_with_outliers.iterrows():
                            col = row['Column']
                            outliers_count = int(row['Outliers Count'])
                            lower = row['Lower Bound']
                            upper = row['Upper Bound']
                            
                            with st.expander(f"📊 {col} - {outliers_count} outliers found", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("📉 Lower Bound", f"{lower:.2f}")
                                with col2:
                                    st.metric("📈 Upper Bound", f"{upper:.2f}")
                                with col3:
                                    st.metric("🔢 Outlier Count", outliers_count)
                                
                                if col in outlier_data:
                                    values = outlier_data[col]['values']
                                    outlier_vals_df = pd.DataFrame({
                                        'Outlier Value': values[:10],
                                        'Status': ['⚠️ Outlier'] * len(values[:10])
                                    })
                                    st.write(f"**First 10 outlier values in `{col}`:**")
                                    st.dataframe(outlier_vals_df, use_column_width=True, hide_index=True)
                                    if len(values) > 10:
                                        st.write(f"... and {len(values) - 10} more outliers")
                                    st.write(f"**Range of outliers:** {min(values):.2f} to {max(values):.2f}")
                                    outlier_pct = (outliers_count / len(df)) * 100
                                    st.write(f"**Percentage of data:** {outlier_pct:.2f}%")
                    else:
                        st.success("✅ No outliers detected in any numeric column.")
                else:
                    st.info("No numeric columns to detect outliers.")
                
                # ============================================================
                # SECTION 5: AI-GENERATED INSIGHTS
                # ============================================================
                st.subheader("🤖 AI-Generated Insights")
                
                if st.button("📊 Generate Insights from Data", use_column_width=True):
                    with st.spinner("🤖 AI is analyzing your data..."):
                        try:
                            response = requests.post(f"{st.session_state.api_url}/explain-data")
                            if response.status_code == 200:
                                result = response.json()
                                if result.get('success'):
                                    st.markdown(f"""
                                    <div class="ai-box">
                                        <h4>📊 Data Insights</h4>
                                        <p style="font-size: 1.05rem; line-height: 1.8;">{result.get('explanation', 'No insights generated')}</p>
                                        <p style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">
                                            Powered by {result.get('llm_provider', 'AI')}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.session_state.insights = result.get('explanation', '')
                                else:
                                    st.info("💡 " + result.get('explanation', 'AI insights not available.'))
                            else:
                                st.error(f"Error: {response.text}")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                # ============================================================
                # SECTION 6: EXECUTIVE SUMMARY
                # ============================================================
                st.divider()
                st.subheader("📄 Executive Summary")

                if st.button("📊 Generate Executive Summary", use_column_width=True):
                    with st.spinner("Generating executive summary..."):
                        # Get cleaned data
                        if st.session_state.cleaned_df is not None:
                            df_clean = st.session_state.cleaned_df
                        else:
                            response = requests.get(f"{st.session_state.api_url}/full-dataset")
                            if response.status_code == 200:
                                full_data = response.json()
                                df_clean = pd.DataFrame(full_data.get('data', []))
                            else:
                                df_clean = df
                        
                        # Build detailed executive summary
                        summary = f"""
# 📊 Executive Summary Report

## 1. Dataset Overview
- **Total Records:** {len(df_clean):,}
- **Total Columns:** {len(df_clean.columns)}
- **Numeric Columns:** {len(df_clean.select_dtypes(include=[np.number]).columns)}
- **Categorical Columns:** {len(df_clean.select_dtypes(include=['object']).columns)}
- **Memory Usage:** {df_clean.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

## 2. Data Quality Metrics
- **Duplicate Rows:** {df_clean.duplicated().sum():,} 
- **Missing Values:** {df_clean.isnull().sum().sum():,}

## 3. Numeric Columns Statistics
"""
                        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
                        for col in numeric_cols:
                            summary += f"""
### {col}
- **Mean:** {df_clean[col].mean():.2f}
- **Median:** {df_clean[col].median():.2f}
- **Min:** {df_clean[col].min():.2f}
- **Max:** {df_clean[col].max():.2f}
- **Std Dev:** {df_clean[col].std():.2f}
"""
                        
                        summary += "\n## 4. Categorical Columns Distribution\n"
                        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
                        for col in categorical_cols[:5]:
                            if len(df_clean[col].value_counts()) > 0:
                                top = df_clean[col].value_counts().index[0]
                                top_count = df_clean[col].value_counts().iloc[0]
                                top_pct = (top_count / len(df_clean) * 100)
                                summary += f"""
### {col}
- **Most Common:** {top}
- **Count:** {top_count:,} ({top_pct:.1f}%)
- **Total Categories:** {df_clean[col].nunique()}
"""
                        
                        if hasattr(st.session_state, 'insights') and st.session_state.insights:
                            summary += f"""
## 5. AI-Generated Insights
{st.session_state.insights}
"""
                        
                        summary += f"""
## 6. Outlier Detection Summary
- **Method Used:** IQR (Interquartile Range)
- **Threshold:** 1.5 × IQR

---
*Report generated by AI Data Analysis Assistant on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
                        
                        st.markdown(summary, unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Centered Download Button
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.download_button(
                                label="📥 Download Executive Summary (Markdown)",
                                data=summary,
                                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown",
                                use_column_width=True
                            )
            else:
                st.error("Could not load dataset for statistics")

# ================================================================
# FOOTER
# ================================================================

st.markdown("""
<div class="footer">
    <p>Built with ❤️ using FastAPI + Streamlit + DeepSeek AI</p>
    <p style="font-size: 0.75rem;">Saylani Hackathon 2026</p>
</div>
""", unsafe_allow_html=True)