import optuna
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def knn_model(X_train, X_test, y_train, y_test, n_trials=10):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    def objective(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 50, 500, step=50)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
        return cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params