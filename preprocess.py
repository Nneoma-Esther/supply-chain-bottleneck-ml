import pandas as pd

def preprocess(path):
    df = pd.read_csv(path)

    df = df.dropna()

    X = df.drop("bottleneck", axis=1)
    y = df["bottleneck"]

    return X, y