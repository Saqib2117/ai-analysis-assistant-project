"""
Visualization Engine for AI Data Analysis Assistant
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ChartGenerator:
    """Generate various charts from dataset"""
    
    def __init__(self, df):
        self.df = df
        self.chart_dir = "charts"
        os.makedirs(self.chart_dir, exist_ok=True)
    
    def _generate_filename(self, prefix="chart"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.chart_dir}/{prefix}_{timestamp}.png"
    
    def generate_chart(self):
        """Auto-generate the most appropriate chart"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determine best chart type
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # Bar chart
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            data = self.df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
            bars = ax.bar(data.index.astype(str), data.values, color=colors)
            
            ax.set_title(f"{num_col} by {cat_col}", fontsize=14, fontweight='bold')
            ax.set_xlabel(cat_col, fontsize=12)
            ax.set_ylabel(num_col, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            
            for bar, value in zip(bars, data.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{value:,.0f}', ha='center', va='bottom', fontsize=9)
        
        elif len(categorical_cols) >= 1:
            # Pie chart
            data = self.df[categorical_cols[0]].value_counts().head(8)
            colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
            ax.pie(data.values, labels=data.index.astype(str), autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax.set_title(f"Distribution of {categorical_cols[0]}", fontsize=14, fontweight='bold')
        
        elif len(numeric_cols) >= 2:
            # Scatter plot
            ax.scatter(self.df[numeric_cols[0]], self.df[numeric_cols[1]], alpha=0.6)
            ax.set_title(f"{numeric_cols[1]} vs {numeric_cols[0]}", fontsize=14, fontweight='bold')
            ax.set_xlabel(numeric_cols[0], fontsize=12)
            ax.set_ylabel(numeric_cols[1], fontsize=12)
        
        else:
            # Histogram
            ax.hist(self.df[numeric_cols[0]], bins=20, color='#2E86AB', edgecolor='black')
            ax.set_title(f"Distribution of {numeric_cols[0]}", fontsize=14, fontweight='bold')
            ax.set_xlabel(numeric_cols[0], fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
        
        plt.tight_layout()
        
        chart_filename = self._generate_filename("chart")
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_filename
    
    def generate_bar_chart(self, x_column, y_column=None):
        """Generate bar chart"""
        if y_column:
            data = self.df.groupby(x_column)[y_column].sum().sort_values(ascending=False)
        else:
            data = self.df[x_column].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
        bars = ax.bar(data.index.astype(str), data.values, color=colors)
        
        ax.set_title(f"Bar Chart: {x_column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(x_column, fontsize=12)
        ax.set_ylabel(y_column or "Count", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        for bar, value in zip(bars, data.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{value:,.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = self._generate_filename("bar")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def generate_pie_chart(self, column):
        """Generate pie chart"""
        data = self.df[column].value_counts().head(8)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        ax.pie(data.values, labels=data.index.astype(str), autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title(f"Pie Chart: {column}", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filename = self._generate_filename("pie")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def generate_histogram(self, column, bins=20):
        """Generate histogram"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.df[column].dropna(), bins=bins, color='#2E86AB', edgecolor='black')
        ax.axvline(self.df[column].mean(), color='red', linestyle='dashed', linewidth=2,
                   label=f'Mean: {self.df[column].mean():.2f}')
        ax.set_title(f"Histogram: {column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        filename = self._generate_filename("histogram")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename
    
    def generate_scatter(self, x_column, y_column):
        """Generate scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.df[x_column], self.df[y_column], alpha=0.6, color='#2E86AB')
        ax.set_title(f"Scatter: {y_column} vs {x_column}", fontsize=14, fontweight='bold')
        ax.set_xlabel(x_column, fontsize=12)
        ax.set_ylabel(y_column, fontsize=12)
        
        plt.tight_layout()
        filename = self._generate_filename("scatter")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename