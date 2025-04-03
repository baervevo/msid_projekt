import numpy as np
import pandas as pd

def get_numerical_columns(df, *force_categorical):
    for column in df.columns:
        if df[column].dtype in [np.int64, np.float64] and not column in force_categorical:
            yield column