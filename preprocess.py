import pandas as pd

def preprocess(path):

    # Load dataset
    df = pd.read_csv(path)

    # Select useful columns
    df = df[[
        "Stock levels",
        "Lead times",
        "Order quantities",
        "Shipping times",
        "Shipping costs",
        "Production volumes",
        "Manufacturing lead time",
        "Defect rates"
    ]]

    # Create target variable
    df["bottleneck"] = (df["Defect rates"] > df["Defect rates"].mean()).astype(int)

    # Features (input variables)
    X = df.drop(["Defect rates", "bottleneck"], axis=1)

    # Target variable
    y = df["bottleneck"]

    return X, y