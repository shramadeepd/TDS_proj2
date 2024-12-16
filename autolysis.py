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
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
         try:
             data = pd.read_csv(file_path, encoding='latin1')
         except Exception as e:
            print(f"Error loading data with multiple encodings: {e}")
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    messages = generate_initial_prompt(file_path)
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
                messages.append({"role": "assistant", "content": result if result else "Function call completed."})
            else:
                 break
    return analysis_steps, data


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
      corr_matrix = data[columns].corr()
      return corr_matrix.to_string()
    except Exception as e:
        return f"Error finding correlation: {e}"


def plot_correlation_matrix(data):
    try:
      numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
      corr_matrix = data[numerical_cols].corr()
      plt.figure(figsize=(10, 8))
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
      plt.title("Correlation Matrix")
      plt.savefig("correlation_matrix.png", bbox_inches='tight')
      plt.close()

    except Exception as e:
        print(f"Error plotting correlation matrix: {e}")


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
      for col in columns:
          if data[col].dtype in [np.int64, np.float64]:
            plt.figure(figsize=(8, 6))
            sns.histplot(data[col], kde=True)
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.savefig(f"histogram_{col}.png", bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"Error plotting histogram: {e}")

def cluster_analysis(data, columns, n_clusters=3):
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        numerical_data = data[columns].copy()
        numerical_data = numerical_data.dropna()

        if numerical_data.empty:
            return "No numerical data available for clustering after dropping NA."

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numerical_data)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        data['cluster'] = kmeans.fit_predict(scaled_features)
        return f"Clustering completed successfully with {n_clusters} clusters."
    except Exception as e:
        return f"Error in cluster analysis: {e}"


def plot_cluster_scatter(data, x_column, y_column):
    try:
        if 'cluster' not in data.columns:
           return "Clustering hasn't been performed yet. Cannot plot cluster scatter."
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=x_column, y=y_column, hue='cluster', data=data, palette='viridis')
        plt.title('Cluster Scatter Plot')
        plt.savefig('cluster_scatter.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting cluster scatter: {e}")

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
    messages = [
            {
                "role": "system",
                 "content": "You are an expert data storyteller, turning analysis results into an engaging narrative. Briefly describe the data, then explain the analysis steps, highlighting key insights and their implications. The analysis steps should be provided in order. Be brief. Keep your response as short as possible. Don't add extra commentary, and avoid using the word 'narrative'. Do not respond with more than 4 paragraphs."
            },
            {
                "role": "user",
                "content": f"Here are the analysis steps:\n{analysis_steps}.\nUse them to generate the story."
            }
        ]

    response = send_to_llm(messages)
    if response:
        return response['content']
    else:
        return "Failed to generate the story."

def save_markdown(analysis, charts_exist):
    markdown_content = f"# Analysis Report\n\n{analysis}\n\n"

    if charts_exist:
        markdown_content += f"\n\n## Visualizations\n"
        for filename in os.listdir("."):
            if filename.endswith(".png"):
                 markdown_content += f"![{filename}]({filename})\n"

    with open("README.md", "w") as f:
            f.write(markdown_content)
    print("Analysis report saved to README.md")


def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    file_path = sys.argv[1]
    analysis_steps, data = load_and_analyze_data(file_path)
    if analysis_steps is None or data is None:
        sys.exit(1)
    analysis = generate_narrative(analysis_steps, data)
    charts_exist = any(filename.endswith(".png") for filename in os.listdir("."))
    save_markdown(analysis, charts_exist)

if __name__ == "__main__":
    main()
