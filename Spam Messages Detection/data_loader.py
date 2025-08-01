import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    return df
