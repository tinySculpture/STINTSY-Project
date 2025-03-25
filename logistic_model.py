import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_logistic_regression(df, features, target, model_path="logistic_model.pkl"):
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # model
    logreg = LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        C=1.0,
        max_iter=500
    )
    logreg.fit(X_scaled, y)

    # save both scaler and model
    joblib.dump({'model': logreg, 'scaler': scaler}, model_path)
    print(f"Model saved to {model_path}")

def load_logistic_model(model_path="logistic_model.pkl"):
    return joblib.load(model_path)
