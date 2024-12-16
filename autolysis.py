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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import json

# Constants
BASE_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"  # Replace with actual base URL

# Function to call LLM API
def call_llm(messages, function_call=None):
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        raise EnvironmentError("AIPROXY_TOKEN environment variable is not set.")
    
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.7,
    }
    if function_call:
        data["function_call"] = function_call

    response = requests.post(BASE_URL, headers=headers, json=data)
    if response.status_code != 200:
        raise ValueError(f"API Error: {response.text}")
    return response.json()

# Function to load CSV and perform initial analysis
def load_and_analyze(csv_file):
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Error reading {csv_file}: {e}")
    
    summary = {
        "shape": data.shape,
        "columns": [{"name": col, "type": str(data[col].dtype)} for col in data.columns],
        "missing": data.isnull().sum().to_dict(),
        "sample": data.head(5).to_dict(),
    }
    return data, summary

# Function to create visualizations
def create_visualizations(data, output_dir):
    sns.set(style="whitegrid")
    charts = []

    # Correlation Heatmap
    if data.select_dtypes(include=["number"]).shape[1] > 1:
        plt.figure(figsize=(10, 8))
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        heatmap_path = f"{output_dir}/correlation_heatmap.png"
        plt.savefig(heatmap_path)
        plt.close()
        charts.append(heatmap_path)

    # Distribution Plot
    for col in data.select_dtypes(include=["number"]).columns[:1]:  # Limit to 1 plot
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True)
        dist_path = f"{output_dir}/distribution_{col}.png"
        plt.savefig(dist_path)
        plt.close()
        charts.append(dist_path)

    return charts

# Function to generate a narrative
def generate_narrative(summary, charts):
    messages = [
        {"role": "system", "content": "You are an analytical assistant creating insights from data."},
        {"role": "user", "content": f"Here's a summary of the dataset: {json.dumps(summary)}."},
        {"role": "user", "content": f"Attached are visualizations: {', '.join(charts)}. Write a story about this analysis."},
    ]
    return call_llm(messages)["choices"][0]["message"]["content"]

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_dir = os.path.splitext(csv_file)[0]
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load and analyze data
    data, summary = load_and_analyze(csv_file)

    # Step 2: Create visualizations
    charts = create_visualizations(data, output_dir)

    # Step 3: Generate narrative
    narrative = generate_narrative(summary, charts)

    # Step 4: Save README.md
    readme_path = f"{output_dir}/README.md"
    with open(readme_path, "w") as readme:
        readme.write(f"# Data Analysis Report\n\n{narrative}\n\n")
        for chart in charts:
            readme.write(f"![{chart}]({chart})\n")

    print(f"Analysis completed. Results saved in {output_dir}.")

if __name__ == "__main__":
    main()
