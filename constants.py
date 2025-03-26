# DEFAULTS

DEFAULT_DATASET_PATH = 'data/dataset.csv'

# COLUMN LABELS

FEATURE = 'feature'
FEATURE_TYPE = 'feature_type'
MEAN = 'mean'
MEDIAN = 'median'
STANDARD_DEVIATION = 'std'
MINIMUM = 'min'
MAXIMUM = 'max'
def PERCENTILE(percentile):
    return f'{percentile}_percentile'
MISSING_VALUES = 'missing_values'
UNIQUE_CLASSES = 'unique_classes'
CLASS_PROPORTIONS = 'class_proportions'

# FEATURE TYPES

NUMERICAL = 'numerical'
CATEGORICAL = 'categorical'