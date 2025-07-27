import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def split_data(df, cfg):
    X = df.drop(columns=[cfg["target_column"]])
    y = df[cfg["target_column"]]
    return train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y)
