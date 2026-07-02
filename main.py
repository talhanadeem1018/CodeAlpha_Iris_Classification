import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
    # -------------------------------
    # Train-Test Split
    # -------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining Features Shape:")
    print(X_train.shape)

    print("\nTesting Features Shape:")
    print(X_test.shape)

    print("\nTraining Labels Shape:")
    print(y_train.shape)

    print("\nTesting Labels Shape:")
    print(y_test.shape)
    
    # -------------------------------
    # Feature Scaling
    # -------------------------------

    scaler = StandardScaler()

    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the testing data
    X_test_scaled = scaler.transform(X_test)

    print("\nScaled Training Features (First 5 Rows):")
    print(X_train_scaled[:5])

    print("\nScaled Testing Features (First 5 Rows):")
    print(X_test_scaled[:5])
    # -------------------------------
    # Train Random Forest Model
    # -------------------------------

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train_scaled, y_train)

    print("\nRandom Forest model trained successfully!")
    
    # -------------------------------
    # Make Predictions
    # -------------------------------
    
    y_pred = model.predict(X_test_scaled)
    
    print("\nActual Labels:")
    print(y_test.values)
    
    print("\nPredicted Labels:")
    print(y_pred)
    
    # -------------------------------
    # Model Accuracy
    # -------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Accuracy:")
    print(f"{accuracy * 100:.2f}%")
    
    # -------------------------------
    # Classification Report
    # -------------------------------

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # -------------------------------
    # Confusion Matrix
    # -------------------------------

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
if __name__ == "__main__":
    main()