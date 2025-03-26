import pandas as pd

def save_to_csv(data, path):
    pd.DataFrame(data).to_csv(path, index=False)

def read_from_csv(path):
    return pd.read_csv(path)