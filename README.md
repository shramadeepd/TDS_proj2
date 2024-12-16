# TDS_proj2
project2 of Tools in Data science 

Problem Explanation:

The core problem is to automate the process of data analysis, visualization, and storytelling using a Large Language Model (LLM). This means creating a system that can take a raw dataset (specifically a CSV file), understand its contents, extract meaningful insights, present those insights visually, and then communicate the findings in a clear and compelling narrative.

Here's a more detailed breakdown:

Data Input:

The system needs to accept a CSV file as input.

The structure and content of the CSV file are unknown in advance. The system should be able to handle diverse datasets without specific pre-configuration.

Data Analysis:

The system must perform generic data analysis techniques that are applicable to a wide range of datasets. This includes:

Calculating descriptive statistics (mean, median, standard deviation, etc.)

Identifying missing values.

Detecting outliers.

Finding correlations between variables.

Potentially performing clustering or other forms of grouping.

The system should dynamically determine the best analysis strategies based on the content of the data.

The LLM should be used to guide the analysis, not to directly perform the computations (LLMs are bad with numbers). The LLM should be prompted to suggest what analysis steps to take based on the data context.

Visualization:

The system must generate visualizations that effectively present the key findings from the analysis.

The system must create 1-3 charts and save them as PNG files.

The type of charts should be appropriate for the data being presented (e.g., histograms for distributions, scatter plots for relationships, correlation matrices as heatmaps).

The LLM can be used to suggest what visualizations to generate, and the system should be able to handle it without breaking if there are errors.

Narrative Generation:

The system must produce a narrative report that summarizes the data, the analysis steps, the key insights, and the implications of the findings.

The narrative must be clear, concise, and understandable by a non-technical audience.

The LLM should be used to generate the narrative based on the analysis results and generated visualizations. The narrative should be in Markdown and should include the visualizations.

LLM Integration:

The core challenge is to effectively leverage an LLM to:

Guide the analysis process (what to analyze and how)

Summarize the analysis results.

Recommend visualizations.

Generate a narrative.

The script needs to minimize token usage with LLM calls and avoid sending large datasets directly to the LLM.

The interactions with the LLM need to be dynamic, using function calling to perform specific actions.

Technical Requirements:

The system must be implemented as a single Python script (autolysis.py).

All dependencies must be included inline in the script.

The system needs to be robust to handle various CSV file structures.

Challenges:

Unknown Data: The script needs to be able to analyze any CSV data without pre-configuration.

LLM Limitations: The LLM is not good with numbers, so analysis computations must be done in Python.

Token Management: Minimize token usage by avoiding sending the entire dataset and using concise prompts.

Dynamic Analysis: The system should adapt its analysis based on the data's content, rather than using static approaches.

Visualization Decisions: The tool should be able to ask the LLM what visualizations would be best.

Function Calling: The tool must be able to perform function calling based on the analysis suggested by the LLM.

Robustness: The system must be able to handle unexpected data, errors in LLM responses, and other issues without crashing.

Output Format: The tool must be able to produce files in the correct structure for the spec.

Desired Outcome:

The final product is a fully automated Python script that can:

Load and analyze any valid CSV file.

Extract meaningful insights using a guided LLM approach.

Generate 1-3 relevant visualizations.

Produce a clear, concise, and insightful narrative report in Markdown with the visualizations embedded.

Successfully communicate all the insights derived.

Be robust and efficient.

In essence, this project aims to build a smart assistant that can take a raw dataset and deliver a ready-to-use analysis report, leveraging the power of LLMs for automation and narrative generation.
