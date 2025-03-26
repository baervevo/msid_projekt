import pandas as pd

def save_to_csv(data, path):
    try:
        pd.DataFrame(data).to_csv(path, index=False)
    except Exception as e:
        print(f'Error: {e}')

def read_from_csv(path, encoding='utf-8'):
    try:
        return pd.read_csv(path, encoding=encoding)
    except FileNotFoundError as e:
        print("File not found!")
        return None