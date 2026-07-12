"""
Data Analysis Engine for AI Data Analysis Assistant
"""

import pandas as pd
import numpy as np
from utils.helpers import get_dataset_summary

class DataAnalyzer:
    """Main data analysis class"""
    
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
    
    def answer_question(self, question, llm_function=None):
        """Answer natural language questions using pandas + LLM fallback"""
        question_lower = question.lower()
        
        # ================================================================
        # 1. HIGHEST / MAXIMUM
        # ================================================================
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
                max_index = self.df[target_col].idxmax()
                row = self.df.loc[max_index]
                
                label_col = None
                for col in self.df.columns:
                    if any(word in col.lower() for word in ['product', 'category', 'name', 'city', 'region', 'type']):
                        label_col = col
                        break
                
                if label_col:
                    return {
                        "answer": f"{row[label_col]} with {target_col} of {max_value:,.2f}",
                        "explanation": f"The highest {target_col} is {max_value:,.2f} from '{row[label_col]}'.",
                        "success": True,
                        "method": "pandas"
                    }
                else:
                    return {
                        "answer": f"{max_value:,.2f}",
                        "explanation": f"The maximum {target_col} is {max_value:,.2f}.",
                        "success": True,
                        "method": "pandas"
                    }
        
        # ================================================================
        # 2. LOWEST / MINIMUM
        # ================================================================
        if "lowest" in question_lower or "minimum" in question_lower or "smallest" in question_lower:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = None
                for col in numeric_cols:
                    if col.lower() in question_lower:
                        target_col = col
                        break
                if not target_col:
                    target_col = numeric_cols[0]
                
                min_value = self.df[target_col].min()
                min_index = self.df[target_col].idxmin()
                row = self.df.loc[min_index]
                
                label_col = None
                for col in self.df.columns:
                    if any(word in col.lower() for word in ['product', 'category', 'name', 'city', 'region', 'type']):
                        label_col = col
                        break
                
                if label_col:
                    return {
                        "answer": f"{row[label_col]} with {target_col} of {min_value:,.2f}",
                        "explanation": f"The lowest {target_col} is {min_value:,.2f} from '{row[label_col]}'.",
                        "success": True,
                        "method": "pandas"
                    }
                else:
                    return {
                        "answer": f"{min_value:,.2f}",
                        "explanation": f"The minimum {target_col} is {min_value:,.2f}.",
                        "success": True,
                        "method": "pandas"
                    }
        
        # ================================================================
        # 3. AVERAGE / MEAN
        # ================================================================
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
                    "answer": f"{avg_value:.2f}",
                    "explanation": f"The average {target_col} is {avg_value:.2f}.",
                    "success": True,
                    "method": "pandas"
                }
        
        # ================================================================
        # 4. SUM / TOTAL
        # ================================================================
        if "sum" in question_lower or "total" in question_lower:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = None
                for col in numeric_cols:
                    if col.lower() in question_lower:
                        target_col = col
                        break
                if not target_col:
                    target_col = numeric_cols[0]
                
                sum_value = self.df[target_col].sum()
                return {
                    "answer": f"{sum_value:,.2f}",
                    "explanation": f"The total {target_col} is {sum_value:,.2f}.",
                    "success": True,
                    "method": "pandas"
                }
        
        # ================================================================
        # 5. STANDARD DEVIATION  <--- FIXED!
        # ================================================================
        if "standard deviation" in question_lower or "std" in question_lower:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = None
                for col in numeric_cols:
                    if col.lower() in question_lower:
                        target_col = col
                        break
                if not target_col:
                    target_col = numeric_cols[0]
                
                std_value = self.df[target_col].std()
                mean_value = self.df[target_col].mean()
                return {
                    "answer": f"{std_value:.2f}",
                    "explanation": f"The standard deviation of {target_col} is {std_value:.2f}. The mean is {mean_value:.2f}.",
                    "success": True,
                    "method": "pandas"
                }
        
        # ================================================================
        # 6. COUNT
        # ================================================================
        if "how many" in question_lower or "count" in question_lower:
            if "unique" in question_lower:
                for col in self.df.columns:
                    if any(word in col.lower() for word in ['product', 'category', 'city']):
                        return {
                            "answer": f"{self.df[col].nunique()}",
                            "explanation": f"There are {self.df[col].nunique()} unique {col}s.",
                            "success": True,
                            "method": "pandas"
                        }
            else:
                return {
                    "answer": f"{len(self.df)}",
                    "explanation": f"The dataset contains {len(self.df)} total records.",
                    "success": True,
                    "method": "pandas"
                }
        
        # ================================================================
        # 7. MOST FREQUENT / MOST COMMON
        # ================================================================
        if "most frequent" in question_lower or "most common" in question_lower:
            for col in self.df.columns:
                if any(word in col.lower() for word in ['category', 'type', 'city', 'status']):
                    top_value = self.df[col].value_counts().index[0]
                    top_count = self.df[col].value_counts().iloc[0]
                    percentage = (top_count / len(self.df)) * 100
                    return {
                        "answer": f"{top_value} ({top_count} occurrences, {percentage:.1f}%)",
                        "explanation": f"The most frequent {col} is '{top_value}' with {top_count} occurrences ({percentage:.1f}%).",
                        "success": True,
                        "method": "pandas"
                    }
        
        # ================================================================
        # 8. MEDIAN
        # ================================================================
        if "median" in question_lower:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = None
                for col in numeric_cols:
                    if col.lower() in question_lower:
                        target_col = col
                        break
                if not target_col:
                    target_col = numeric_cols[0]
                
                median_value = self.df[target_col].median()
                return {
                    "answer": f"{median_value:.2f}",
                    "explanation": f"The median {target_col} is {median_value:.2f}.",
                    "success": True,
                    "method": "pandas"
                }
        
        # ================================================================
        # 9. LLM FALLBACK (if pandas can't answer)
        # ================================================================
        if llm_function:
            try:
                columns_info = ", ".join(self.df.columns.tolist())
                sample_data = self.df.head(5).to_string()
                
                prompt = f"""
                You are a data analysis assistant. Answer the following question based on the dataset.
                
                Dataset Columns: {columns_info}
                Sample Data (first 5 rows):
                {sample_data}
                
                Question: {question}
                
                Provide a concise answer and a brief explanation.
                Format: ANSWER: <answer> | EXPLANATION: <explanation>
                """
                
                llm_response = llm_function(prompt)
                
                if llm_response:
                    if "ANSWER:" in llm_response and "EXPLANATION:" in llm_response:
                        parts = llm_response.split("EXPLANATION:")
                        answer_part = parts[0].replace("ANSWER:", "").strip()
                        explanation_part = parts[1].strip() if len(parts) > 1 else ""
                        
                        return {
                            "answer": answer_part,
                            "explanation": explanation_part,
                            "success": True,
                            "method": "llm"
                        }
                    else:
                        return {
                            "answer": llm_response[:200],
                            "explanation": "Generated by AI",
                            "success": True,
                            "method": "llm"
                        }
            except Exception as e:
                print(f"LLM fallback error: {e}")
        
        # ================================================================
        # DEFAULT
        # ================================================================
        return {
            "answer": "I couldn't determine the answer from the dataset.",
            "explanation": "Please rephrase your question or check the dataset columns.",
            "success": False,
            "method": "none"
        }