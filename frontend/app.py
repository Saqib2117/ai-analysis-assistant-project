"""
AI Data Analysis Assistant - Premium UI
Deployed Backend: https://saqib21-fastapi-backend.hf.space
"""

import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO

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
# CUSTOM CSS (Premium UI)
# ================================================================

st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Answer Boxes */
    .answer-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .explanation-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.3rem 0;
    }
    
    /* Chat Bubbles */
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 5px 18px;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    .assistant-msg {
        background: #f1f3f5;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 5px;
        max-width: 70%;
        margin-right: auto;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    
    /* AI Explanation Box */
    .ai-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# SESSION STATE
# ================================================================

if 'api_url' not in st.session_state:
    st.session_state.api_url = "https://saqib21-fastapi-backend.hf.space"
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'chart_path' not in st.session_state:
    st.session_state.chart_path = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'llm_available' not in st.session_state:
    st.session_state.llm_available = False
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "None"
if 'api_key_error' not in st.session_state:
    st.session_state.api_key_error = None

# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 4rem;">📊</div>
        <h2 style="color: #667eea;">AI Assistant</h2>
        <p style="color: #6c757d; font-size: 0.9rem;">Data Analysis Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # LLM Status
    llm_status = "✅ Connected" if st.session_state.llm_available else "⚠️ LLM Not Configured"
    llm_color = "#28a745" if st.session_state.llm_available else "#ffc107"
    st.markdown(f"""
    <div style="background: #e8f4fd; padding: 0.8rem; border-radius: 10px; text-align: center;">
        <span style="font-size: 1.2rem;">🤖</span>
        <span style="font-weight: 600; color: #667eea;">DeepSeek AI</span>
        <br>
        <span style="font-size: 0.8rem; color: {llm_color};">{llm_status}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <h4 style="color: #667eea; margin-bottom: 0.5rem;">📤 Upload Dataset</h4>
    <p style="color: #6c757d; font-size: 0.85rem;">Upload a CSV file to analyze</p>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.info(f"📄 {uploaded_file.name} ({uploaded_file.size/1024:.1f} KB)")
        
        if st.button("🚀 Upload & Analyze", use_container_width=True):
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
                        
                        # Auto-generate chart
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
    <h4 style="color: #667eea; margin-bottom: 0.5rem;">✨ Features</h4>
    """, unsafe_allow_html=True)
    
    features = [
        "📊 Auto Charts",
        "❓ Natural Language Q&A",
        "🤖 AI Explanations",
        "📄 PDF Export",
        "🎨 Dark Mode",
        "📥 Chart Download"
    ]
    for f in features:
        st.markdown(f"<p style='font-size: 0.9rem;'>• {f}</p>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown(f"""
    <p style="font-size: 0.8rem; color: #6c757d; text-align: center;">
        🔗 Backend: Hugging Face<br>
        ⚡ Status: <span style="color: #28a745;">● Online</span>
    </p>
    """, unsafe_allow_html=True)

# ================================================================
# MAIN CONTENT
# ================================================================

st.markdown('<h1 class="main-header">📊 AI Data Analysis Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your CSV and ask questions in natural language</p>', unsafe_allow_html=True)

# ================================================================
# MAIN CONTENT AREA
# ================================================================

if not st.session_state.data_loaded:
    # Welcome Screen - More Colorful
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem;">
        <div style="font-size: 5rem;">🚀</div>
        <h2 style="color: #667eea; margin-top: 0.5rem;">AI Data Analysis Assistant</h2>
        <p style="color: #6c757d; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Upload your CSV file to get started with AI-powered analysis
        </p>
        <div style="margin-top: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 2rem; border-radius: 15px; max-width: 500px; margin-left: auto; margin-right: auto;">
            <p style="font-weight: 600; color: #667eea;">📌 Quick Start Guide</p>
            <p style="text-align: left; font-size: 0.95rem;">
                1️⃣ Click <strong>"Upload Dataset"</strong> in the sidebar<br>
                2️⃣ Select a CSV file from your computer<br>
                3️⃣ View automatic dataset summary<br>
                4️⃣ Ask questions about your data<br>
                5️⃣ Generate professional charts
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    summary = st.session_state.summary
    
    # Metrics Row
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
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
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
            st.image(st.session_state.chart_path, use_container_width=True)
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
                st.markdown(f"""
                <div class="assistant-msg">
                    <strong>🤖 AI:</strong><br>{item['text']}
                    <br><small style="color: #6c757d;">{item.get('explanation', '')}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        col1, col2 = st.columns([5, 1])
        
        with col1:
            question = st.text_input(
                "Ask a question",
                placeholder="e.g., Which product generated the highest sales?",
                key="question_input"
            )
        
        with col2:
            ask_button = st.button("❓ Ask", use_container_width=True)
        
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
                if st.button(q, key=f"q_{i}", use_container_width=True):
                    st.session_state.question_input = q
                    st.rerun()
    
    # ================================================================
    # TAB 3: Charts (FIXED - with AI Explanation)
    # ================================================================
    
    with tab3:
        st.subheader("📈 Chart Generation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Auto", "Bar", "Pie", "Histogram", "Scatter"]
            )
        
        with col2:
            st.write("")
            if st.button("🎨 Generate Chart", use_container_width=True):
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
                            st.image(chart_path, use_container_width=True)
                            st.success("✅ Chart generated!")
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Display current chart
        if st.session_state.chart_path:
            st.image(st.session_state.chart_path, use_container_width=True)
            
            with open(st.session_state.chart_path, "rb") as f:
                st.download_button(
                    label="📥 Download Chart (PNG)",
                    data=f,
                    file_name="chart.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        # --- AI Chart Explanation Section (FIXED) ---
        st.divider()
        st.subheader("🤖 AI Chart Explanation")
        
        if not st.session_state.llm_available:
            st.warning("⚠️ **DeepSeek API Key Not Configured on Backend.**")
            st.markdown("""
            **To enable AI explanations:**
            1. Go to your Hugging Face Space → **Settings** → **Variables and secrets**
            2. Click **"New secret"**
            3. **Name:** `DEEPSEEK_API_KEY`
            4. **Value:** Your DeepSeek API key (starts with `sk-`)
            5. Click **"Add secret"**
            6. Restart your Space
            """)
        
        if st.button("📊 Explain This Chart with AI", use_container_width=True):
            with st.spinner("🤖 AI is analyzing your chart..."):
                try:
                    response = requests.post(f"{st.session_state.api_url}/explain-chart")
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            st.markdown(f"""
                            <div class="ai-box">
                                <h4 style="color: #667eea;">🤖 AI Analysis</h4>
                                <p style="font-size: 1.05rem; line-height: 1.8;">{result.get('explanation', 'No explanation generated')}</p>
                                <p style="color: #6c757d; font-size: 0.85rem; margin-top: 0.5rem;">
                                    Powered by {result.get('llm_provider', 'AI')}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(result.get('explanation', '⚠️ LLM not configured. Please add DEEPSEEK_API_KEY as a Secret on Hugging Face.'))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ================================================================
    # TAB 4: AI Explain (FIXED - with DeepSeek Setup Instructions)
    # ================================================================
    
    with tab4:
        st.subheader("🤖 AI Chart Explanation")
        st.markdown("""
        <p style="color: #6c757d;">
            Get AI-powered insights and explanations about your data and charts.
        </p>
        """, unsafe_allow_html=True)
        
        if not st.session_state.llm_available:
            st.warning("⚠️ **DeepSeek API Key Not Configured on Backend.**")
            st.markdown("""
            ### 🔑 How to Add DeepSeek API Key to Hugging Face:
            
            1. Go to your Hugging Face Space:  
               `https://huggingface.co/spaces/saqib21/fastapi-backend/settings`
            
            2. Scroll to **"Variables and secrets"**
            
            3. Click **"New secret"**
            
            4. **Name:** `DEEPSEEK_API_KEY`
            
            5. **Value:** Your DeepSeek API key (starts with `sk-`)
            
            6. Click **"Add secret"**
            
            7. **Restart your Space** (Settings → Restart this Space)
            
            8. Wait for rebuild (~2 minutes)
            
            9. Refresh this page!
            """)
        
        if st.button("📊 Explain My Data with AI", use_container_width=True):
            with st.spinner("🤖 AI is analyzing your data..."):
                try:
                    response = requests.post(f"{st.session_state.api_url}/explain-chart")
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('success'):
                            st.markdown(f"""
                            <div class="ai-box">
                                <h4 style="color: #667eea;">🤖 AI Analysis</h4>
                                <p style="font-size: 1.05rem; line-height: 1.8;">{result.get('explanation', 'No explanation generated')}</p>
                                <p style="color: #6c757d; font-size: 0.85rem; margin-top: 0.5rem;">
                                    Powered by {result.get('llm_provider', 'AI')}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(result.get('explanation', '⚠️ LLM not configured. Please add DEEPSEEK_API_KEY as a Secret on Hugging Face.'))
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
            if st.button("📊 Export PDF Report", use_container_width=True):
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
                                    use_container_width=True
                                )
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("📋 Export Summary (JSON)", use_container_width=True):
                st.json(st.session_state.summary)

# ================================================================
# FOOTER
# ================================================================

st.markdown("""
<div class="footer">
    <p>Built with ❤️ using FastAPI + Streamlit + DeepSeek AI</p>
    <p style="font-size: 0.8rem;">Saylani Hackathon 2026</p>
</div>
""", unsafe_allow_html=True)