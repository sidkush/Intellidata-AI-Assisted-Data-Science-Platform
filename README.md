# Intellidata-AI-Assisted-Data-Science-Platform

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.11 or higher
- Access to the internet for downloading required packages

## Installation

Follow these steps to install the necessary libraries and set up the project:

1. Clone the repository (if applicable):
git clone https://github.com/sidkush/Intellidata-AI-Assisted-Data-Science-Platform.git

2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate

3. Install the required libraries:
pip install -r requirements.txt

4. Put your API key in string format in `api_key.py` file

## Running the Code

To run the application, use the following command:
streamlit run app.py
or python -m streamlit run "c:/Users/[path]/app.py"

## Usage

Once the application is running, you can interact with the various features provided by the AI Assistance for Data Science tool. The interface is designed to be intuitive, guiding you through the data science tasks you wish to perform.

## Business Problem Statement

This project addresses the need for an accessible, user-friendly tool that guides users through the data science process. It aims to democratize data science by providing AI assistance for exploratory data analysis (EDA), data analysis, and machine learning tasks.

## Methodology

1. **Data Upload**: Users begin by uploading a CSV file.
2. **Exploratory Data Analysis (EDA)**: Automated EDA is performed on the uploaded dataset.
3. **Targeted Data Analysis**: Users can investigate specific variables of interest.
4. **Data Science Problem Framing**: The app helps users convert business problems into data science problems.
5. **Algorithm Selection**: Based on the problem, the app suggests suitable machine learning algorithms.
6. **Code Generation**: The app can generate Python code to solve the defined problem using the selected algorithm.

## Project Structure
Intellidata-AI-Assisted-Data-Science-Platform
│
├── app.py # Main Streamlit application
├── api_key.py # File containing OpenAI API key
├── requirements.txt # Required libraries
└── README.md # Project documentation

## Key Features

### 1. Interactive UI with Streamlit

The application uses Streamlit to create an interactive web interface. Users can upload data, trigger analyses, and receive AI-generated insights through a user-friendly dashboard.

### 2. AI-Powered Assistance

The app leverages OpenAI's language models to provide intelligent assistance throughout the data science process. It uses:

- `ChatOpenAI`: For generating human-like responses and explanations.
- `create_pandas_dataframe_agent`: For performing operations on the uploaded dataset.
- `create_python_agent`: For generating Python code to solve data science problems.

### 3. Workflow Stages

#### a. Exploratory Data Analysis (EDA)

The EDA process includes:
- Data overview (column meanings, missing values, duplicates)
- Data summarization (outlier detection, feature suggestions)

#### b. Targeted Data Analysis

Users can input specific variables for detailed analysis, including:
- Summary statistics
- Normality checks
- Outlier assessment
- Trend analysis

#### c. Data Science Problem Solving

The app guides users through:
- Converting business problems to data science problems
- Suggesting appropriate machine learning algorithms
- Generating Python code for the selected algorithm

### 4. Wikipedia Integration

The app uses the `WikipediaAPIWrapper` to fetch relevant information, enhancing the context for algorithm suggestions.

### 5. Conversation Memory

The app maintains a conversation history, allowing for contextual interactions with the AI assistant.

## Solution and Approach

This application solves the challenge of making data science accessible to users with varying levels of expertise. It does so by:

1. **Guiding Through the Process**: The app breaks down the data science workflow into manageable steps, from data upload to problem-solving.

2. **Automating Complex Tasks**: By leveraging AI, the app automates tasks like EDA and algorithm selection, which typically require significant expertise.

3. **Providing Contextual Information**: The app offers explanations and best practices at each step, educating users as they progress.

4. **Generating Code**: For more advanced users, the app can generate Python code to implement solutions, bridging the gap between concept and implementation.

## Example Usage

Let's walk through a hypothetical use case:

1. A user uploads a CSV file containing customer data for an e-commerce platform.

2. They click the "EDA" button, and the app provides insights about the dataset structure, missing values, and potential features of interest.

3. The user then asks for a detailed analysis of the "Purchase Amount" variable. The app provides summary statistics, checks for normality, and identifies any trends or seasonality in purchase amounts.

4. Next, the user inputs a business problem: "How can we predict which customers are likely to make a purchase in the next month?"

5. The app converts this into a data science problem, suggesting it's a binary classification task.

6. It then recommends suitable algorithms like Logistic Regression, Random Forest, and Gradient Boosting.

7. The user selects Random Forest, and the app generates Python code to implement this solution using the uploaded dataset.

Throughout this process, the user can interact with the AI assistant for clarifications or additional insights, making the entire data science journey more approachable and educational.
