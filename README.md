# 📊 AI Data Analysis Assistant

An AI-powered data analysis assistant that helps you understand your CSV data through natural language queries, automated visualizations, and smart insights.

---

## 📖 Overview

The **AI Data Analysis Assistant** is a full-stack web application that allows users to upload CSV datasets and interact with their data using natural language. It combines FastAPI for backend processing, Streamlit for an interactive frontend, and DeepSeek LLM for intelligent data insights.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📤 **CSV Upload** | Upload any CSV file with automatic error handling |
| 📊 **Auto Analysis** | Instantly generates dataset summary, statistics, and data types |
| 💬 **Natural Language Q&A** | Ask questions like "Which product has the highest sales?" |
| 📈 **Smart Charts** | Automatically generates the most appropriate chart based on your data |
| 🤖 **AI Insights** | DeepSeek LLM provides intelligent explanations about your data |
| 📄 **PDF Reports** | Export professional, branded analysis reports |
| 🌙 **Dark Mode** | Toggle between light and dark themes |
| 📥 **Chart Download** | Download charts as PNG images |

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Matplotlib** | Chart generation |
| **Seaborn** | Statistical visualizations |
| **ReportLab** | PDF report generation |
| **DeepSeek API** | LLM for insights |

### Frontend
| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web UI framework |
| **Requests** | API communication |
| **Plotly** | Interactive charts |

### Deployment
| Platform | Purpose |
|----------|---------|
| **Huggingface** | Backend hosting |
| **Streamlit Cloud** | Frontend hosting |

---

## 🔄 Project Workflow

1. **Upload CSV** → User selects CSV file from their computer
2. **Auto Analysis** → System generates dataset summary and statistics
3. **Ask Questions** → User types questions in natural language
4. **Generate Charts** → System creates visualizations based on data
5. **AI Explanation** → DeepSeek provides insights about the data
6. **Export Report** → User downloads PDF report with all findings

---

## 🌐 Live Demo

### Backend API (Deployed on Render)
 - Backend Deployed on Render, here is the deployed URL: https://saqib21-fastapi-backend.hf.space 

 ### Frontend (Deployed on Streamlit)
 - Frontend of this project is deployed on streamlit, here is the deployed url: https://ai-analysis-assistant-project.streamlit.app/
