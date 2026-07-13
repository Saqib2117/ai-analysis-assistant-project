"""
Data Analysis Engine for AI Data Analysis Assistant
AI-First Approach: LLM understands the dataset and answers questions
"""

import pandas as pd
import numpy as np
import json
from utils.helpers import get_dataset_summary

class DataAnalyzer:
    """Main data analysis class - AI-First"""
    
    def __init__(self):
        self.df = None
        self.summary = None
        self.is_loaded = False
    
    def load_data(self, df):
        self.df = df
        self.summary = get_dataset_summary(df)
        self.is_loaded = True
        return True
    
    def get_summary(self):
        if not self.is_loaded:
            return {"error": "No data loaded"}
        return self.summary
    
    def get_basic_info(self):
        if not self.is_loaded:
            return {"error": "No data loaded"}
        
        info = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "head": self.df.head(10).to_dict('records'),
            "describe": self.df.describe().to_dict() if len(self.df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        return info
    
    def _format_result(self, result, question):
        """Format the result for human readability - CLEAN & CONCISE"""
        
        # Clean the question
        clean_question = question.replace("?", "").strip()
        
        # If result is a Series (like groupby result) - COMPLEX ANSWER
        if isinstance(result, pd.Series):
            sorted_result = result.sort_values(ascending=False)
            
            formatted = f"📊 {clean_question}\n\n"
            formatted += "| Category | Value |\n"
            formatted += "|----------|-------|\n"
            
            for idx, val in sorted_result.items():
                if isinstance(val, (int, float)):
                    formatted += f"| {idx} | {val:,.2f} |\n"
                else:
                    formatted += f"| {idx} | {val} |\n"
            
            if len(sorted_result) > 0:
                max_val = sorted_result.max()
                max_idx = sorted_result.idxmax()
                min_val = sorted_result.min()
                min_idx = sorted_result.idxmin()
                formatted += f"\n\n💡 Highest: {max_idx} ({max_val:,.2f}) | Lowest: {min_idx} ({min_val:,.2f})"
            
            return formatted
        
        # If result is a number - SHORT ANSWER WITH FULL SENTENCE
        elif isinstance(result, (int, float)):
            # Build a complete sentence
            # Extract the key from the question
            question_lower = question.lower()
            subject = ""
            if "age" in question_lower:
                subject = "age"
            elif "hours" in question_lower or "hour" in question_lower:
                subject = "hours per week"
            elif "gain" in question_lower:
                subject = "capital gain"
            elif "loss" in question_lower:
                subject = "capital loss"
            elif "income" in question_lower:
                subject = "income"
            else:
                # Try to find the column name in the question
                for col in self.df.columns:
                    if col.lower() in question_lower:
                        subject = col
                        break
                if not subject:
                    subject = "value"
            
            return f"The average {subject} in the dataset is {result:,.2f}."
        
        # If result is a string
        elif isinstance(result, str):
            return result
        
        # If result is a DataFrame
        elif isinstance(result, pd.DataFrame):
            return result.to_string(index=False)
        
        # Default
        return str(result)
    
    def _get_dataset_context(self):
        """Get comprehensive dataset context for LLM"""
        columns_info = ", ".join(self.df.columns.tolist())
        data_types = self.df.dtypes.to_dict()
        sample_data = self.df.head(10).to_string()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        stats_info = ""
        for col in numeric_cols[:5]:
            stats_info += f"- {col}: min={self.df[col].min():.2f}, max={self.df[col].max():.2f}, mean={self.df[col].mean():.2f}\n"
        
        cat_info = ""
        for col in categorical_cols[:3]:
            top = self.df[col].value_counts().head(3)
            cat_info += f"- {col}: {dict(top)}\n"
        
        context = f"""
DATASET INFORMATION:
- Columns: {columns_info}
- Numeric Columns: {numeric_cols}
- Categorical Columns: {categorical_cols}
- Total Rows: {len(self.df)}

NUMERIC STATISTICS (first 5 columns):
{stats_info}

CATEGORICAL TOP VALUES (first 3 columns):
{cat_info}

SAMPLE DATA:
{sample_data}
"""
        return context
    
    def _is_simple_question(self, question):
        """Check if the question is a simple statistic question"""
        simple_patterns = [
            "average", "mean", "maximum", "minimum", "highest", "lowest",
            "total", "sum", "count", "how many", "most frequent", "most common",
            "median", "standard deviation"
        ]
        question_lower = question.lower()
        # Check if it's asking for a single value (not "for each" or "by")
        if "for each" in question_lower or " by " in question_lower or " per " in question_lower:
            return False
        for pattern in simple_patterns:
            if pattern in question_lower:
                return True
        return False
    
    def answer_question(self, question, llm_function=None):
        """Answer using LLM to understand and generate pandas code"""
        
        if llm_function:
            try:
                context = self._get_dataset_context()
                
                prompt = f"""
You are an expert data analyst AI. You have a dataset loaded in pandas DataFrame called 'df'.

{context}

USER QUESTION: "{question}"

Your task is to:
1. UNDERSTAND what the user is asking
2. WRITE the correct pandas code to answer the question
3. EXECUTE the code and return the answer

Follow these rules:
- Use ONLY pandas operations (groupby, mean, max, min, sum, count, value_counts, etc.)
- Store the final answer in a variable called 'result'
- Keep the code SIMPLE and EFFICIENT
- Handle edge cases (empty results, missing columns, etc.)

IMPORTANT: If the result is a Series (like groupby), keep it as a Series.
The final answer should be stored in 'result'.

EXAMPLE 1: "What is the average age?"
CODE: result = df['age'].mean()

EXAMPLE 2: "What is the average hours per week for each workclass?"
CODE: result = df.groupby('workclass')['hours-per-week'].mean()

EXAMPLE 3: "How many people earn more than 50K?"
CODE: result = len(df[df['income'] == '>50K'])

EXAMPLE 4: "What is the most common occupation?"
CODE: result = df['occupation'].value_counts().index[0]

Now, generate ONLY the pandas code for this question.
Do NOT include any explanation, just the code.
The code should store the answer in a variable called 'result'.

Your response should be ONLY the Python code, nothing else.
"""
                
                llm_response = llm_function(prompt)
                
                if llm_response:
                    code = llm_response.strip()
                    
                    if code.startswith('```python'):
                        code = code.replace('```python', '').replace('```', '').strip()
                    elif code.startswith('```'):
                        code = code.replace('```', '').strip()
                    
                    local_vars = {'df': self.df, 'pd': pd, 'np': np}
                    
                    try:
                        exec(code, {}, local_vars)
                        result = local_vars.get('result')
                        
                        if result is not None:
                            formatted_answer = self._format_result(result, question)
                            
                            return {
                                "answer": formatted_answer,
                                "explanation": "",
                                "success": True,
                                "method": "llm_pandas"
                            }
                    
                    except Exception as e:
                        print(f"Code execution error: {e}")
                        direct_prompt = f"""
The user asked: "{question}"
The dataset has columns: {', '.join(self.df.columns.tolist())}
The dataset has {len(self.df)} rows.

Please answer the question directly.
If it's a simple question (average, max, min, count), return a complete sentence.
If it's a complex question (group by), return the data in a clear format.
"""
                        direct_answer = llm_function(direct_prompt)
                        if direct_answer:
                            return {
                                "answer": direct_answer,
                                "explanation": "",
                                "success": True,
                                "method": "llm_direct"
                            }
            
            except Exception as e:
                print(f"LLM-first approach error: {e}")
        
        return self._fallback_answer(question, llm_function)
    
    def _fallback_answer(self, question, llm_function=None):
        """Simple fallback for basic questions"""
        question_lower = question.lower()
        
        try:
            # Highest / Maximum
            if "highest" in question_lower or "maximum" in question_lower or "largest" in question_lower:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_col = None
                    for col in numeric_cols:
                        if col.lower() in question_lower:
                            target_col = col
                            break
                    if not target_col:
                        target_col = numeric_cols[0]
                    
                    max_value = self.df[target_col].max()
                    return {
                        "answer": f"The maximum {target_col} in the dataset is {max_value:,.2f}.",
                        "explanation": "",
                        "success": True,
                        "method": "fallback"
                    }
            
            # Average / Mean
            if "average" in question_lower or "mean" in question_lower:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_col = None
                    for col in numeric_cols:
                        if col.lower() in question_lower:
                            target_col = col
                            break
                    if not target_col:
                        target_col = numeric_cols[0]
                    
                    avg_value = self.df[target_col].mean()
                    return {
                        "answer": f"The average {target_col} in the dataset is {avg_value:,.2f}.",
                        "explanation": "",
                        "success": True,
                        "method": "fallback"
                    }
            
            # ================================================================
            # MOST FREQUENT CATEGORY - IMPROVED VERSION (UPDATED)
            # ================================================================
            if "most frequent" in question_lower or "most common" in question_lower:
                cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
                
                if not cat_cols:
                    return {
                        "answer": "No categorical columns found in the dataset.",
                        "explanation": "The dataset only contains numeric data.",
                        "success": False,
                        "method": "pandas"
                    }
                
                # Check if user specified a column
                target_col = None
                for col in cat_cols:
                    if col.lower() in question_lower:
                        target_col = col
                        break
                
                if target_col:
                    # Answer for specific column
                    top = self.df[target_col].value_counts().index[0]
                    count = self.df[target_col].value_counts().iloc[0]
                    pct = (count / len(self.df)) * 100
                    return {
                        "answer": f"The most frequent {target_col} is '{top}' with {count} occurrences ({pct:.1f}%).",
                        "explanation": "",
                        "success": True,
                        "method": "pandas"
                    }
                else:
                    # List all categorical columns with their most frequent values
                    result = "Most frequent values by column:\n\n"
                    for col in cat_cols[:10]:
                        top = self.df[col].value_counts().index[0]
                        count = self.df[col].value_counts().iloc[0]
                        pct = (count / len(self.df)) * 100
                        result += f"• {col}: {top} ({pct:.1f}%)\n"
                    
                    if len(cat_cols) > 10:
                        result += f"\n... and {len(cat_cols) - 10} more columns"
                    
                    result += f"\n\n💡 Tip: Ask 'What is the most frequent [column_name]?' for a specific column."
                    
                    return {
                        "answer": result,
                        "explanation": "",
                        "success": True,
                        "method": "pandas"
                    }
            
            # Count
            if "how many" in question_lower or "count" in question_lower:
                return {
                    "answer": f"The dataset contains {len(self.df)} total records.",
                    "explanation": "",
                    "success": True,
                    "method": "fallback"
                }
        
        except Exception as e:
            print(f"Fallback error: {e}")
        
        if llm_function:
            try:
                direct_prompt = f"""
The user asked: "{question}"
Dataset columns: {', '.join(self.df.columns.tolist())}
Dataset rows: {len(self.df)}

Provide a complete sentence answer.
"""
                direct_answer = llm_function(direct_prompt)
                if direct_answer:
                    return {
                        "answer": direct_answer,
                        "explanation": "",
                        "success": True,
                        "method": "llm_fallback"
                    }
            except:
                pass
        
        return {
            "answer": "I couldn't determine the answer from the dataset.",
            "explanation": "Please rephrase your question or check the dataset columns.",
            "success": False,
            "method": "none"
        }