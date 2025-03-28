import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target variable

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale features

    return X_scaled, y, scaler  # Return scaler for later use


def objective(trial, X_train_scaled, y_train):
    n_neighbors = trial.suggest_int('n_neighbors', 50, 500, step=50)  # Optimize k
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
    return cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()


def createstudy(X_train_scaled, y_train, n_trials=10):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train), n_trials=n_trials)
    return study.best_params


def train_best_knn(X_train_scaled, y_train, best_params):
    best_k = best_params['n_neighbors']
    model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    
    print("\n" + classification_report(y_test, y_pred) + "\n")
    print(confusion_matrix(y_test, y_pred))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return y_pred
