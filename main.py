import pandas as pd

print("TEST: Python is running the correct file")

def main():
    print("=" * 50)
    print("CodeAlpha - Iris Flower Classification")
    print("=" * 50)

    df = pd.read_csv("dataset/Iris.csv")

    print("\nDataset Loaded Successfully!\n")
    print(df.head())
    # Dataset Shape
    print("\nDataset Shape:")
    print(df.shape)

    # Column Names
    print("\nColumn Names:")
    print(df.columns)

    # Dataset Information
    print("\nDataset Information:")
    df.info()

    # Missing Values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Statistical Summary
    print("\nStatistical Summary:")
    print(df.describe())

    # Species Count
    print("\nSpecies Count:")
    print(df["Species"].value_counts())
    
    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    
    # Remove Id column
    df = df.drop("Id", axis=1)
    
    # Features (X)
    X = df.drop("Species", axis=1)
    
    # Target (y)
    y = df["Species"]
    
    print("\nFeatures (X):")
    print(X.head())
    
    print("\nTarget (y):")
    print(y.head())
if __name__ == "__main__":
    main()