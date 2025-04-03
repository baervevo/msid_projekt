# Project Part 1: Data Analysis

This project focuses on analyzing datasets using Python scripts. Below is an overview of the project structure and usage instructions.

These scripts are intended for use with the Estimation of Obesity Levels Based On Eating Habits and Physical Condition dataset

https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

## Project Structure
- `base_statistics.py`: Contains functions for basic statistical calculations.
- `data_analysis_main.py`: Main script for performing data analysis using the provided dataset.
- `requirements.txt`: Lists the Python dependencies required for the project.

## Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd projekt_czesc_1
    ```

2. **Set Up Virtual Environment**:
    Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run `data_analysis_main.py`**:
    Execute the main script to analyze your dataset:
    ```bash
    python data_analysis_main.py
    ```

2. **Use `base_statistics.py`**:
    Calculate the basic statistics for each feature in the chosen dataset and output them into a file.
    ```bash
    python base_statistics.py <input_dataset> <output_file>
    ```

## Notes
- Ensure your dataset is in the correct format before running the scripts.
- Activate the virtual environment before executing any Python scripts.
