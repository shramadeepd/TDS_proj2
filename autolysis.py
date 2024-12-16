# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas>=2.0.0",
#     "numpy>=1.24.0", 
#     "seaborn>=0.12.0",
#     "matplotlib>=3.7.0",
#     "requests>=2.31.0",
#     "python-dotenv>=1.0.0",
#     "scikit-learn>=1.3.0"
# ]
# ///

import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv
from collections import Counter
import time
import logging

load_dotenv()

# Load the AI Proxy token from environment variable
API_KEY = os.environ.get("AIPROXY_TOKEN")
if not API_KEY:
    raise ValueError("AIPROXY_TOKEN environment variable not set.")

BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"

# Function to send a message to the LLM and get a response
def send_to_llm(messages, function_call=None, functions=None, max_retries=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "function_call": function_call,
        "functions": functions
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            response_data = response.json()
            if response_data and "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]
            else:
                logging.warning(f"LLM Response: Unexpected format, attempt {attempt + 1}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during LLM communication: {e}, attempt {attempt + 1}")
            time.sleep(1)
    return None


# Function to extract function call name and arguments from a message
def extract_function_call(message):
    if message and 'function_call' in message and message['function_call']:
        return message['function_call']['name'], json.loads(message['function_call']['arguments'])
    return None, None

# Function to execute function calls
def execute_function_call(name, arguments, data):
    if name == "describe_columns":
        return describe_columns(data)
    elif name == "generate_summary_stats":
        return generate_summary_stats(data, **arguments)
    elif name == "find_correlation":
        return find_correlation(data, **arguments)
    elif name == "plot_correlation_matrix":
        plot_correlation_matrix(data)
        return "Correlation matrix plot generated successfully"
    elif name == "detect_outliers":
        return detect_outliers(data, **arguments)
    elif name == "plot_histogram":
        plot_histogram(data, **arguments)
        return "Histogram plot generated successfully"
    elif name == "cluster_analysis":
        return cluster_analysis(data, **arguments)
    elif name == "plot_cluster_scatter":
        plot_cluster_scatter(data, **arguments)
        return "Cluster scatter plot generated successfully"
    elif name == "analyze_text_column":
        return analyze_text_column(data, **arguments)
    else:
        return "Function call not recognized."

def generate_initial_prompt(file_path):
        return [
            {
                "role": "system",
                "content": "You are an expert data analyst. Analyze the data and suggest further steps. I will provide a data file path. Based on its content and structure, tell me what analysis steps to take. Keep it brief."
            },
            {
                "role": "user",
                "content": f"I have a dataset in a CSV file named {file_path}. The file contains a variety of data. Please provide a detailed analysis plan, including specific function calls (use the functions I have defined), and the information I should send you with the function calls. I have a limited amount of tokens to do all this, so keep it brief. Keep your responses as short as possible. Do not explain or comment in your responses."
            }
        ]


# Functions the LLM can call:
functions = [
    {
        "name": "describe_columns",
        "description": "Get information about the columns in the dataset.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    },
     {
        "name": "generate_summary_stats",
        "description": "Generate summary statistics for numerical columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "description": "List of numerical column names to analyze.",
                    "items": {"type": "string"}
                }
            },
            "required": ["columns"]
        },
    },
    {
        "name": "find_correlation",
        "description": "Calculate the correlation matrix for numerical columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "description": "List of numerical columns to analyze",
                    "items": {"type": "string"}
                }
            },
            "required": ["columns"]
        }
    },
    {
        "name": "plot_correlation_matrix",
        "description": "Generate and save a heatmap of the correlation matrix.",
         "parameters": {
            "type": "object",
            "properties": {
            },
        }
    },
    {
        "name": "detect_outliers",
        "description": "Identify outliers in numerical columns based on IQR.",
          "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "description": "List of numerical columns to check for outliers.",
                    "items": {"type": "string"}
                }
            },
            "required": ["columns"]
        }
    },
     {
        "name": "plot_histogram",
        "description": "Generates and saves histograms for numerical columns.",
          "parameters": {
            "type": "object",
             "properties": {
                "columns": {
                    "type": "array",
                    "description": "List of numerical columns to plot.",
                    "items": {"type": "string"}
                }
            },
             "required": ["columns"]
        }
    },
    {
        "name": "cluster_analysis",
        "description": "Performs clustering analysis on specified columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "description": "List of numerical columns to perform clustering on.",
                    "items": {"type": "string"}
                },
                "n_clusters": {
                     "type": "integer",
                    "description": "Number of clusters to form.",
                     "default": 3
                }
            },
            "required": ["columns"]
        }
    },
    {
        "name": "plot_cluster_scatter",
         "description": "Generates and saves a scatter plot of the clusters.",
        "parameters": {
             "type": "object",
             "properties": {
                 "x_column":{
                   "type": "string",
                   "description":"Column for X axis"
                  },
                   "y_column":{
                    "type": "string",
                   "description":"Column for Y axis"
                  }
            },
            "required": ["x_column", "y_column"]
        }
    },
    {
        "name": "analyze_text_column",
        "description": "Analyze a specific text column.",
        "parameters": {
            "type": "object",
             "properties": {
                "column": {
                   "type": "string",
                   "description":"Text column to analyze."
                  }
            },
            "required": ["column"]
        }
    }
]

def load_and_analyze_data(file_path):
    """Load and analyze the dataset."""
    if not file_path or os.path.isdir(file_path):
        logging.error(f"Invalid file path or directory: {file_path}")
        return None, None

    try:
        try:
            data = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, encoding='latin1')

        logging.info(f"Successfully loaded dataset with {len(data)} rows and {len(data.columns)} columns")
        if data.empty:
            logging.error("Dataset is empty.")
            return None, None

        messages = generate_initial_prompt(file_path)
        analysis_steps = []
        
        for _ in range(3):
            try:
                message = send_to_llm(messages, functions=functions)
                if not message:
                    continue

                messages.append(message)
                function_name, function_args = extract_function_call(message)

                if function_name:
                    result = execute_function_call(function_name, function_args, data)
                    if result:
                        analysis_steps.append(f"Function call: {function_name} with arguments {function_args}, Result: {result}")
                    messages.append({"role": "assistant", "content": result if result else "Function call completed."})
                else:
                    break
            except Exception as e:
                logging.error(f"Error in analysis loop: {e}")
                continue
        return analysis_steps, data

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None


def describe_columns(data):
    column_info = []
    for col in data.columns:
        col_type = str(data[col].dtype)
        example_value = str(data[col].iloc[0]) if not data[col].empty else 'NA'
        unique_values = len(data[col].unique())
        missing_values = data[col].isnull().sum()
        column_info.append(f"Column: {col}, Type: {col_type}, Example Value: {example_value}, Unique Values: {unique_values}, Missing Values: {missing_values}")
    return "\n".join(column_info)


def generate_summary_stats(data, columns):
     try:
        summary = data[columns].describe().to_string()
        return summary
     except Exception as e:
         return f"Error generating summary statistics: {e}"


def find_correlation(data, columns):
    try:
        valid, error_msg = validate_columns(data, columns)
        if not valid:
            return error_msg
        
        non_numeric = [col for col in columns if not np.issubdtype(data[col].dtype, np.number)]
        if non_numeric:
            return f"Non-numeric columns found: {', '.join(non_numeric)}"

        corr_matrix = data[columns].corr()
        return corr_matrix.to_string()
    except Exception as e:
        return f"Error finding correlation: {e}"


def plot_correlation_matrix(data):
    try:
        output_dir = os.path.splitext(os.path.basename(sys.argv[1]))[0]
        os.makedirs(output_dir, exist_ok=True)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            logging.warning("Not enough numeric columns for correlation matrix")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        
        output_path = os.path.join(output_dir, "correlation_matrix.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Saved correlation matrix to {output_path}")
    except Exception as e:
        logging.error(f"Error plotting correlation matrix: {e}")


def detect_outliers(data, columns):
    try:
        outliers = {}
        for col in columns:
            if data[col].dtype in [np.int64, np.float64]:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                outliers[col] = col_outliers.to_list()
        return json.dumps(outliers)
    except Exception as e:
         return f"Error finding outliers: {e}"


def plot_histogram(data, columns):
    try:
        valid, error_msg = validate_columns(data, columns)
        if not valid:
            logging.error(error_msg)
            return

        output_dir = os.path.splitext(os.path.basename(sys.argv[1]))[0]
        os.makedirs(output_dir, exist_ok=True)
            
        for col in columns:
            if not np.issubdtype(data[col].dtype, np.number):
                logging.warning(f"Column {col} is not numeric, skipping histogram")
                continue

            plt.figure(figsize=(8, 6))
            sns.histplot(data=data[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")

            output_path = os.path.join(output_dir, f"histogram_{col}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Saved histogram for {col} to {output_path}")

    except Exception as e:
        logging.error(f"Error plotting histograms: {e}")


def cluster_analysis(data, columns, n_clusters=3):
    try:
        valid, error_msg = validate_columns(data, columns)
        if not valid:
            return error_msg
            
        non_numeric = [col for col in columns if not np.issubdtype(data[col].dtype, np.number)]
        if non_numeric:
            return f"Non-numeric columns found: {', '.join(non_numeric)}"
            
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        numerical_data = data[columns].copy()
        numerical_data = numerical_data.dropna()

        if numerical_data.empty:
            return "No data available for clustering after dropping NA values."

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numerical_data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        data['cluster'] = kmeans.fit_predict(scaled_features)
        return f"Clustering completed successfully with {n_clusters} clusters."
    except Exception as e:
        return f"Error in cluster analysis: {e}"


def plot_cluster_scatter(data, x_column, y_column):
    try:
        output_dir = os.path.splitext(os.path.basename(sys.argv[1]))[0]
        if 'cluster' not in data.columns:
            return "Clustering hasn't been performed yet. Cannot plot cluster scatter."
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=x_column, y=y_column, hue='cluster', data=data, palette='viridis')
        plt.title('Cluster Scatter Plot')
        output_path = os.path.join(output_dir, 'cluster_scatter.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved cluster scatter plot to {output_path}")
    except Exception as e:
        logging.error(f"Error plotting cluster scatter: {e}")


def analyze_text_column(data, column):
    try:
        text_data = data[column].dropna()
        if text_data.empty:
            return f"No text data found in column: {column} after dropping NA."
        
        all_text = ' '.join(text_data.astype(str).tolist())
        words = all_text.lower().split()
        word_counts = Counter(words)

        most_common_words = word_counts.most_common(10)
        return json.dumps(most_common_words)
    except Exception as e:
        return f"Error analyzing text column: {e}"


def generate_narrative(analysis_steps, data):
    """Generate narrative and visualizations based on LLM suggestions."""
    messages = [
        {
            "role": "system",
            "content": """You are an expert data analyst. First analyze the data and suggest visualizations, 
            then create a narrative. Respond in this exact format:
            
            VISUALIZATIONS:
            - List specific visualization commands to run (e.g., plot_correlation_matrix, plot_histogram)
            - Include specific columns for histograms
            
            NARRATIVE:
            Write a brief analysis story (max 4 paragraphs)"""
        },
        {
            "role": "user",
            "content": f"Here are the analysis steps:\n{analysis_steps}.\nSuggest visualizations and create the story."
        }
    ]

    response = send_to_llm(messages)
    if not response:
        return "Failed to generate analysis"

    content = response.get('content', '') if isinstance(response, dict) else response
    
    try:
        viz_section, narrative_section = content.split("NARRATIVE:", 1)
        viz_lines = [line.strip() for line in viz_section.split("VISUALIZATIONS:")[1].split("\n") if line.strip()]
        
        for viz in viz_lines:
            if "plot_correlation_matrix" in viz.lower():
                plot_correlation_matrix(data)
            elif "plot_histogram" in viz.lower():
                cols = [col.strip() for col in viz.split("(")[-1].split(")")[0].split(",")]
                plot_histogram(data, cols)
            elif "plot_cluster_scatter" in viz.lower():
                cols = [col.strip() for col in viz.split("(")[-1].split(")")[0].split(",")]
                if len(cols) >= 2:
                    plot_cluster_scatter(data, cols[0], cols[1])
                    
        return narrative_section.strip()
    except Exception as e:
        logging.error(f"Error processing LLM response: {e}")
        return content


def save_markdown(analysis, charts_exist):
    output_dir = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    markdown_content = f"# Analysis Report\n\n{analysis}\n\n"
    
    if charts_exist:
        markdown_content += "\n## Visualizations\n\n"
        for filename in os.listdir("."):
            if filename.endswith(".png"):
                markdown_content += f"![{filename}]({filename})\n\n"
                try:
                    os.rename(filename, os.path.join(output_dir, filename))
                except Exception as e:
                    logging.error(f"Error moving {filename}: {e}")

    try:
        with open(os.path.join(output_dir, "README.md"), "w", encoding='utf-8') as f:
            f.write(markdown_content)
        logging.info(f"Analysis saved to {output_dir}/README.md")
    except Exception as e:
        logging.error(f"Error saving README.md: {e}")


def validate_columns(data, columns):
    if not isinstance(columns, list):
        columns = [columns]
    
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        return False, f"Columns not found in data: {', '.join(missing_columns)}"
    return True, None


def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_dir, exist_ok=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(output_dir, 'autolysis.log')), logging.StreamHandler()]
    )
    try:
        analysis_steps, data = load_and_analyze_data(file_path)
        if analysis_steps is None or data is None:
            sys.exit(1)
            
        analysis = generate_narrative(analysis_steps, data)
        charts_exist = any(filename.endswith(".png") for filename in os.listdir("."))
        save_markdown(analysis, charts_exist)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
