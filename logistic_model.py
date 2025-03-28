from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_features(X_train, X_test):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        C=1.0,
        class_weight='balanced',
        max_iter=500  
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
