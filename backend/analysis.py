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
    
    def _format_result(self, result, question, label=None):
        """Format the result for human readability - PLAIN TEXT, NO MARKDOWN.

        FIXED: previously this always said "The average {subject} in the
        dataset is X" no matter what statistic was actually computed (max,
        sum, count, median, etc.), which is why answers to non-average
        questions still read as "average". It also guessed at the column
        name via keyword matching, so unmatched/complex phrasing silently
        fell back to a made-up "value" label. Now it uses the `label`
        (the actual computed metric, provided by the LLM alongside the code,
        e.g. "average age", "maximum hours-per-week", "count of records
        where income is >50K") so the wording always matches what was
        actually computed. Markdown symbols (#, **, bullets, emoji, pipe
        tables) are removed in favor of clean plain text.
        """

        # If result is a Series (like groupby result) - DETAILED ANSWER (light markdown for readability)
        if isinstance(result, pd.Series):
            sorted_result = result.sort_values(ascending=False)

            heading = label or question.replace("?", "").strip()
            formatted = f"**{heading.capitalize()}:**\n\n"

            for idx, val in sorted_result.items():
                if isinstance(val, (int, float, np.integer, np.floating)):
                    formatted += f"- **{idx}**: {val:,.2f}\n"
                else:
                    formatted += f"- **{idx}**: {val}\n"

            if len(sorted_result) > 0:
                max_val = sorted_result.max()
                max_idx = sorted_result.idxmax()
                min_val = sorted_result.min()
                min_idx = sorted_result.idxmin()
                formatted += f"\nHighest: **{max_idx}** ({max_val:,.2f}). Lowest: **{min_idx}** ({min_val:,.2f})."

            return formatted.strip()

        # If result is a number - SHORT, PLAIN ANSWER (no markdown)
        elif isinstance(result, (int, float, np.integer, np.floating)):
            metric = label or "value"
            if isinstance(result, (int, np.integer)):
                value_str = f"{result:,}"
            else:
                value_str = f"{result:,.2f}"
            return f"The {metric} in the dataset is: {value_str}"

        # If result is a string - SHORT, PLAIN ANSWER WITH LABEL (no markdown)
        elif isinstance(result, str):
            metric = label or question.replace("?", "").strip()
            return f"The {metric} in the dataset is: {result.strip()}"

        # If result is a DataFrame
        elif isinstance(result, pd.DataFrame):
            return result.to_string(index=False)

        # Default
        return str(result)
    
    def _get_dataset_context(self):
        """Get comprehensive dataset context for LLM.

        FIXED: previously only sent stats for the first 5 numeric columns and
        top values for the first 3 categorical columns, so the LLM had no
        grounding for the rest of the dataset and would guess/hallucinate on
        anything outside that narrow slice, especially for complex or
        differently-phrased questions. Now sends stats for every column
        (capped generously) and puts an explicit AVAILABLE COLUMNS list up
        front so the LLM can reliably detect when a question refers to a
        column that isn't actually in the dataset.
        """
        columns_info = ", ".join(self.df.columns.tolist())
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        sample_data = self.df.head(5).to_string()

        stats_info = ""
        for col in numeric_cols[:20]:
            stats_info += (
                f"- {col}: min={self.df[col].min():.2f}, max={self.df[col].max():.2f}, "
                f"mean={self.df[col].mean():.2f}, median={self.df[col].median():.2f}\n"
            )

        cat_info = ""
        for col in categorical_cols[:15]:
            top = self.df[col].value_counts().head(5)
            cat_info += f"- {col} (unique values: {self.df[col].nunique()}): {dict(top)}\n"

        context = f"""
AVAILABLE COLUMNS (this is the COMPLETE list — no other columns exist in this dataset):
{columns_info}

Numeric Columns: {numeric_cols}
Categorical Columns: {categorical_cols}
Total Rows: {len(self.df)}

NUMERIC STATISTICS:
{stats_info}

CATEGORICAL TOP VALUES:
{cat_info}

SAMPLE DATA (first 5 rows):
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
        """Answer using the LLM to read the dataset itself and generate pandas code.

        FIXED:
        - The LLM now also returns a RESULT_LABEL describing what it actually
          computed (e.g. "average age", "maximum hours-per-week"), so the
          final phrasing always matches the real statistic instead of always
          saying "average".
        - The LLM is given the FULL column list and is explicitly told to
          report when a question references a column that doesn't exist,
          instead of silently substituting a similar column.
        - Complex/grouped results now get a short, detailed, plain-text
          explanation generated by the LLM (no markdown symbols), while
          simple single-value questions stay short and direct.
        """

        if not llm_function:
            return self._fallback_answer(question, llm_function)

        try:
            context = self._get_dataset_context()

            prompt = f"""
You are an expert data analyst AI. You have a dataset loaded in a pandas DataFrame called 'df'.

{context}

USER QUESTION: "{question}"

Your task:
1. Check the question against the AVAILABLE COLUMNS list above. Only those columns exist — do not assume, invent, or substitute a similar-sounding column.
2. If the question refers to a column that is NOT in the AVAILABLE COLUMNS list, do not compute anything. Instead respond with exactly:
RESULT_LABEL: COLUMN_NOT_FOUND
CODE:
result = None
3. Otherwise, write simple pandas code using ONLY the given 'df' (groupby, mean, max, min, sum, count, value_counts, etc.) that correctly answers the question, and store the final answer in a variable called 'result'.
4. Also provide a short RESULT_LABEL describing exactly what statistic 'result' represents, in plain lowercase words including the real column name(s), e.g. "average age", "maximum hours-per-week", "count of records where income is >50K", "average hours-per-week by workclass".

Respond in EXACTLY this format, nothing else, no markdown, no explanation:
RESULT_LABEL: <short description of the statistic>
CODE:
result = <pandas code>

EXAMPLE 1 — Question: "What is the average age?"
RESULT_LABEL: average age
CODE:
result = df['age'].mean()

EXAMPLE 2 — Question: "What is the average hours per week for each workclass?"
RESULT_LABEL: average hours-per-week by workclass
CODE:
result = df.groupby('workclass')['hours-per-week'].mean()

EXAMPLE 3 — Question: "How many people earn more than 50K?"
RESULT_LABEL: count of records where income is >50K
CODE:
result = len(df[df['income'] == '>50K'])

EXAMPLE 4 — Question: "What is the most common occupation?"
RESULT_LABEL: most common occupation
CODE:
result = df['occupation'].value_counts().index[0]

EXAMPLE 5 — Question: "What is the sales in the dataset?" (when 'sales' is not an available column)
RESULT_LABEL: COLUMN_NOT_FOUND
CODE:
result = None
"""

            llm_response = llm_function(prompt)

            if llm_response:
                label, code = self._parse_labeled_code(llm_response)

                if label == "COLUMN_NOT_FOUND" or code is None:
                    return self._missing_column_answer(question)

                local_vars = {'df': self.df, 'pd': pd, 'np': np}

                try:
                    exec(code, {}, local_vars)
                    result = local_vars.get('result')

                    if result is not None:
                        formatted_answer = self._format_result(result, question, label=label)

                        # For complex/grouped results, add a short detailed
                        # plain-text explanation as requested for complex questions.
                        if isinstance(result, (pd.Series, pd.DataFrame)):
                            detail = self._generate_detail(question, formatted_answer, llm_function)
                            if detail:
                                formatted_answer = f"{formatted_answer}\n\n{detail}"

                        return {
                            "answer": formatted_answer,
                            "explanation": "",
                            "success": True,
                            "method": "llm_pandas"
                        }
                    else:
                        return self._missing_column_answer(question)

                except KeyError:
                    # The generated code referenced a column that doesn't exist
                    return self._missing_column_answer(question)

                except Exception as e:
                    print(f"Code execution error: {e}")
                    return self._llm_direct_answer(question, llm_function)

        except Exception as e:
            print(f"LLM-first approach error: {e}")

        return self._fallback_answer(question, llm_function)

    def _parse_labeled_code(self, llm_response):
        """Parse the RESULT_LABEL / CODE response from the LLM."""
        text = llm_response.strip()
        if text.startswith('```python'):
            text = text.replace('```python', '').replace('```', '').strip()
        elif text.startswith('```'):
            text = text.replace('```', '').strip()

        label = None
        code = None

        if "RESULT_LABEL:" in text and "CODE:" in text:
            label_part = text.split("RESULT_LABEL:", 1)[1].split("CODE:", 1)[0].strip()
            code_part = text.split("CODE:", 1)[1].strip()
            label = label_part.strip()
            code = code_part.strip()
        else:
            # LLM didn't follow the format — treat the whole thing as code
            code = text

        return label, code

    def _missing_column_answer(self, question):
        """Plain-text response for when the question refers to a column
        that isn't in the dataset."""
        available = ", ".join(self.df.columns.tolist())
        return {
            "answer": (
                f"The dataset doesn't contain a column matching that request. "
                f"Available columns are: {available}"
            ),
            "explanation": "",
            "success": False,
            "method": "column_not_found"
        }

    def _generate_detail(self, question, formatted_answer, llm_function):
        """Generate a short, detailed explanation for complex (grouped/
        multi-value) results. Light markdown (bold on key terms/numbers) is
        allowed here for readability; no headers (#) or bullet-heavy walls of text."""
        try:
            detail_prompt = f"""
The user asked: "{question}"

Here is the computed result:
{formatted_answer}

Write a short, detailed explanation (2-4 sentences) of what this result shows and the key insight or pattern in it.
You may bold key terms, category names, or numbers using **double asterisks** for readability.
Do not use headers (#), bullet points, or emoji. Write as flowing prose, not a list.
"""
            detail = llm_function(detail_prompt)
            if detail:
                # Strip only headers/bullets/emoji; keep ** bold markers intact
                text = detail.strip()
                for symbol in ["##", "#", "•", "📊", "💡", "✅", "⚠️", "■"]:
                    text = text.replace(symbol, "")
                lines = [line.lstrip("- ").rstrip() if line.strip().startswith("- ") else line for line in text.split("\n")]
                return "\n".join(lines).strip()
        except Exception as e:
            print(f"Detail generation error: {e}")
        return None

    def _strip_markdown(self, text):
        """Remove common markdown symbols so responses render as clean plain text."""
        for symbol in ["**", "##", "#", "•", "📊", "💡", "✅", "⚠️", "■"]:
            text = text.replace(symbol, "")
        # Collapse leftover markdown list dashes like "- " at line starts
        lines = [line.lstrip("- ").rstrip() if line.strip().startswith("- ") else line for line in text.split("\n")]
        return "\n".join(lines).strip()

    def _llm_direct_answer(self, question, llm_function):
        """Ask the LLM to answer directly in plain language when pandas code
        generation/execution didn't work out."""
        direct_prompt = f"""
The user asked: "{question}"
The dataset has these columns ONLY: {', '.join(self.df.columns.tolist())}
The dataset has {len(self.df)} rows.

If the question refers to a column not in that list, say clearly that the dataset doesn't contain that column and list the available columns instead.
Otherwise answer the question directly and accurately based on the dataset.
If it's a simple question (average, max, min, count), answer in one short plain sentence in the form: "The <metric> in the dataset is: <value>".
If it's a complex question, give a short detailed plain-text answer (2-4 sentences).
Do not use markdown symbols such as #, *, **, bullet points, or emoji.
"""
        direct_answer = llm_function(direct_prompt)
        if direct_answer:
            return {
                "answer": self._strip_markdown(direct_answer.strip()),
                "explanation": "",
                "success": True,
                "method": "llm_direct"
            }
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
                        result += f"- {col}: {top} ({pct:.1f}%)\n"
                    
                    if len(cat_cols) > 10:
                        result += f"\n... and {len(cat_cols) - 10} more columns"
                    
                    result += f"\n\nTip: Ask 'What is the most frequent [column_name]?' for a specific column."
                    
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