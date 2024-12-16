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
import csv
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv
from collections import Counter
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

load_dotenv()

# Load the AI Proxy token from environment variable
API_KEY = os.environ.get("AIPROXY_TOKEN")
if not API_KEY:
    raise ValueError("AIPROXY_TOKEN environment variable not set.")

BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"  # Specific model for AI Proxy

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
            response = requests.post(BASE_URL, headers=headers, json=payload, timeout=10)  # Added timeout for faster responses
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()
            if response_data and "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]
            else:
                print(f"LLM Response: Unexpected format, attempt {attempt + 1}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error during LLM communication: {e}, attempt {attempt + 1}")
            time.sleep(1) # Add a short delay before retry
    return None

# Function to extract function call name and arguments from a message
def extract_function_call(message):
    if message and 'function_call' in message and message['function_call']:
        return message['function_call']['name'], json.loads(message['function_call']['arguments'])
    return None, None


def execute_function_call(name, arguments, data):
    """Executes function calls based on the provided name and arguments."""
    if name == "describe_data":
        return _describe_data(data)
    elif name == "analyze_numeric_column":
        return _analyze_numeric_column(data, **arguments)
    elif name == "analyze_text_column":
        return _analyze_text_column(data, **arguments)
    elif name == "cluster_data":
         return _cluster_data(data, **arguments)
    elif name == "visualize_data":
        return _visualize_data(data, **arguments)
    else:
        return "Function call not recognized."

def _describe_data(data):
    """Provides a description of the dataset, including column types and missing values."""
    column_info = []
    for col in data.columns:
        col_type = str(data[col].dtype)
        example_value = str(data[col].iloc[0]) if not data[col].empty else 'NA'
        unique_values = len(data[col].unique())
        missing_values = data[col].isnull().sum()
        column_info.append(f"Column: {col}, Type: {col_type}, Example Value: {example_value}, Unique Values: {unique_values}, Missing Values: {missing_values}")
    return "\n".join(column_info)

def _analyze_numeric_column(data, column):
    """Analyzes a numerical column, calculating summary statistics, outliers, and correlations."""
    try:
        if not np.issubdtype(data[column].dtype, np.number):
            return f"Column {column} is not numeric."
            
        summary = data[column].describe()
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column].to_list()
        
        correlation_info = {}
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) > 1:
            correlations = data[numeric_cols].corr()
            correlation_info = correlations[column].drop(column).to_dict() # Drop self correlation

        return {
            "summary": summary.to_dict(),
            "outliers": outliers,
            "correlations": correlation_info
        }
    except Exception as e:
        return f"Error analyzing numeric column: {e}"

def _analyze_text_column(data, column):
    """Analyzes a specific text column."""
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

def _cluster_data(data, columns, n_clusters=3):
    """Performs clustering analysis on specified columns."""
    try:
        # Validate columns exist
        valid, error_msg = validate_columns(data, columns)
        if not valid:
            return error_msg
            
        # Check if columns are numeric
        non_numeric = [col for col in columns if not np.issubdtype(data[col].dtype, np.number)]
        if non_numeric:
            return f"Non-numeric columns found: {', '.join(non_numeric)}"
            
        
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

def _visualize_data(data, plot_type, columns=None, x_column=None, y_column=None):
    """Generates visualizations based on the plot type."""
    output_dir = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        if plot_type == "histogram" and columns:
            for col in columns:
                if np.issubdtype(data[col].dtype, np.number):
                    plt.figure(figsize=(8,6))
                    sns.histplot(data=data[col].dropna(), kde=True)
                    plt.title(f"Distribution of {col}")
                    plt.xlabel(col)
                    plt.ylabel("Frequency")
                    plt.savefig(os.path.join(output_dir, f"histogram_{col}.png"), bbox_inches='tight')
                    plt.close()
                else:
                    print(f"Column {col} is not numeric, skipping histogram")
            return "Histograms generated."
        elif plot_type == "scatter" and x_column and y_column:
            if 'cluster' not in data.columns:
               return "Clustering hasn't been performed yet. Cannot plot cluster scatter."
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=x_column, y=y_column, hue='cluster', data=data, palette='viridis')
            plt.title('Cluster Scatter Plot')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.savefig(os.path.join(output_dir, 'cluster_scatter.png'), bbox_inches='tight')
            plt.close()
            return "Scatter plot generated."
        elif plot_type == "correlation":
            numeric_cols = data.select_dtypes(include=np.number).columns
            if len(numeric_cols) < 2:
                return "Not enough numeric columns for correlation matrix."
            plt.figure(figsize=(10, 8))
            sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Correlation Matrix")
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), bbox_inches='tight', dpi=300)
            plt.close()
            return "Correlation matrix plot generated."
        else:
            return f"Plot type {plot_type} not supported"
    except Exception as e:
        return f"Error during plotting: {e}"


def generate_initial_prompt(file_path, data):
    """Generates the initial prompt for the LLM."""
    data_description = _describe_data(data)
    return [
        {
            "role": "system",
            "content": "You are an expert data analyst. Analyze the provided dataset and suggest what steps to take. I will provide a data file path and a summary of the data. I have defined a set of functions you can call."
        },
        {
            "role": "user",
            "content": f"I have a dataset in a CSV file named {file_path}.\n\n Here is the data description:\n{data_description}\n\nPlease suggest the next step for analysis, focusing on what functions to call and which arguments to use. The goal is to generate insights and visualizations. Be very concise."
        }
    ]

def generate_analysis_prompt(file_path, analysis_steps):
    """Generates a prompt for the LLM to create a narrative."""
    return [
        {
            "role": "system",
            "content": """You are an expert data analyst. Summarize the provided analysis steps into a short narrative. Also, mention any visualizations that were generated."""
        },
        {
            "role": "user",
             "content": f"Here are the analysis steps:\n{analysis_steps}.\n\nCreate a narrative of maximum 4 paragraphs, also mention any visualizations that were created."
         }
    ]

# Functions the LLM can call:
functions = [
    {
    "name": "describe_data",
    "description": "Provide a description of the dataset including column types and missing values.",
    "parameters": {
        "type": "object",
        "properties": {}
        }
    },
    {
        "name": "analyze_numeric_column",
        "description": "Analyze a numerical column, calculating summary statistics, outliers, and correlations.",
          "parameters": {
            "type": "object",
             "properties": {
                "column": {
                   "type": "string",
                   "description":"Numerical column to analyze."
                  }
             },
             "required": ["column"]
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
    },
    {
        "name": "cluster_data",
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
        "name": "visualize_data",
        "description": "Generate a specific plot type.",
         "parameters": {
            "type": "object",
            "properties": {
                "plot_type": {
                   "type": "string",
                   "description":"The type of plot to generate (e.g., 'histogram', 'scatter', 'correlation')."
                 },
                 "columns": {
                     "type": "array",
                     "description":"Columns to use for the plot.",
                     "items": {"type": "string"}
                 },
                  "x_column":{
                   "type": "string",
                   "description":"Column for X axis (for scatter plots)"
                   },
                   "y_column":{
                    "type": "string",
                   "description":"Column for Y axis (for scatter plots)"
                   }
            },
            "required": ["plot_type"]
        }
    }
]


def load_and_analyze_data(file_path):
    """Loads, analyzes, and visualizes data from a CSV file."""
    if os.path.isdir(file_path):
        print(f"Error: {file_path} is a directory. Please provide a CSV file path.")
        return None, None
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            print(f"Error loading data with multiple encodings: {e}")
            return None, None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    messages = generate_initial_prompt(file_path, data)
    analysis_steps = []
    
    while True:
        message = send_to_llm(messages, functions=functions)
        if not message:
            break

        messages.append(message)
        function_name, function_args = extract_function_call(message)

        if function_name:
            result = execute_function_call(function_name, function_args, data)
            if result:
                analysis_steps.append(f"Function call: {function_name} with arguments {function_args}, Result: {result}")
            messages.append({"role": "assistant", "content": str(result) if result else "Function call completed."})
        else:
            break

    narrative_prompt = generate_analysis_prompt(file_path, analysis_steps)
    narrative_response = send_to_llm(narrative_prompt)
    narrative = narrative_response.get('content', '') if narrative_response else "Failed to generate narrative"
    return analysis_steps, narrative, data


def save_markdown(analysis, file_path):
    """Saves the analysis report to a markdown file."""
    output_dir = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Create markdown content
    markdown_content = f"# Analysis Report\n\n{analysis}\n\n"
    
    # Handle images
    markdown_content += "\n## Visualizations\n\n"
    
    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
           markdown_content += f"![{filename}]({filename})\n\n"
    
    # Save README.md in output directory
    try:
        with open(os.path.join(output_dir, "README.md"), "w", encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Analysis saved to {output_dir}/README.md")
    except Exception as e:
        print(f"Error saving README.md: {e}")

def validate_columns(data, columns):
    """Validates that all columns exist in the dataframe."""
    if not isinstance(columns, list):
        columns = [columns]
    
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        return False, f"Columns not found in data: {', '.join(missing_columns)}"
    return True, None


# Function definitions remain largely the same

# Main function
def main():
    """Main entry point of the script."""
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    file_path = sys.argv[1]
    analysis_steps, analysis, data = load_and_analyze_data(file_path)
    if analysis_steps is None or data is None:
         sys.exit(1)
    save_markdown(analysis, file_path)
    

if __name__ == "__main__":
    main()
