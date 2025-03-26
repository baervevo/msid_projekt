import sys
import os
import pandas as pd
import numpy as np

from constants import NUMERICAL, CATEGORICAL, PERCENTILE, FEATURE, FEATURE_TYPE, MEDIAN, MEAN, STANDARD_DEVIATION, MINIMUM, MAXIMUM, MISSING_VALUES, UNIQUE_CLASSES, CLASS_PROPORTIONS, DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_FOLDER
from io_operations import save_to_csv, read_from_csv

def get_percentile(dataset, column, percentile):
    return np.percentile(dataset[column].dropna(), percentile)

def get_missing_values(dataset, column):
    return dataset[column].isna().sum()

def calculate_statistics(dataset):
    statistics = []
    for column in dataset.columns:
        if dataset[column].dtype in [np.int64, np.float64]:
            statistics.append({
                FEATURE: column,
                FEATURE_TYPE: NUMERICAL,
                MEDIAN: dataset[column].mean(),
                MEAN: dataset[column].median(),
                STANDARD_DEVIATION: dataset[column].std(),
                MINIMUM: dataset[column].min(),
                MAXIMUM: dataset[column].max(),
                PERCENTILE(5): get_percentile(dataset, column, 5),
                PERCENTILE(95): get_percentile(dataset, column, 95),
                MISSING_VALUES: get_missing_values(dataset, column)
            })
        else:
            value_counts = dataset[column].value_counts(normalize=True)
            statistics.append({
                FEATURE: column,
                FEATURE_TYPE: CATEGORICAL,
                UNIQUE_CLASSES: dataset[column].nunique(),
                MISSING_VALUES: get_missing_values(dataset, column),
                CLASS_PROPORTIONS: value_counts.to_dict()
            })
    return statistics

def main():
    if len(sys.argv) != 3:
        dataset_path = DEFAULT_DATASET_PATH
        output_path = os.path.join(DEFAULT_OUTPUT_FOLDER, 'statistics.csv')
    else:
        dataset_path = sys.argv[1]
        output_path = sys.argv[2]
    dataset = read_from_csv(dataset_path)
    save_to_csv(calculate_statistics(dataset), output_path)

if __name__ == '__main__':
    main()