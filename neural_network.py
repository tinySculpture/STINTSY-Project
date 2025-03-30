import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes, list_hidden, activation='relu'):
        super(NeuralNetwork, self).__init__()
        self.activation = activation
        layers = []
        
        layers.append(nn.Linear(input_size, list_hidden[0]))
        layers.append(self.get_activation_layer())
        
        for i in range(len(list_hidden) - 1):
            layers.append(nn.Linear(list_hidden[i], list_hidden[i+1]))
            layers.append(self.get_activation_layer())
        
        layers.append(nn.Linear(list_hidden[-1], num_classes))
        
        self.model = nn.Sequential(*layers)
        self.init_weights()
    
    def get_activation_layer(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")
    
    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            return torch.argmax(outputs, dim=1)
    
    def predict_proba(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            return F.softmax(outputs, dim=1)

def train_neural_network(X_train, y_train, input_size, list_hidden, num_classes, activation='relu', epochs=100, learning_rate=0.01, verbose=True):
    set_seed()
    model = NeuralNetwork(input_size, num_classes, list_hidden, activation)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred_tensor = model(X_test_tensor)
    y_pred = torch.argmax(y_pred_tensor, axis=1).numpy()
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred)) 
    
    return accuracy, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column]).values  
    y = df[target_column].values  
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def objective(trial, X_train, y_train, input_size, num_classes):
    set_seed()
    list_hidden = [trial.suggest_int(f'hidden_{i}', 16, 128) for i in range(trial.suggest_int('num_layers', 1, 3))]
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
    epochs = trial.suggest_int('epochs', 50, 200)
    
    model = train_neural_network(X_train, y_train, input_size, list_hidden, num_classes, activation, epochs, learning_rate, verbose=False)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    y_pred_tensor = model(X_train_tensor)
    y_pred = torch.argmax(y_pred_tensor, axis=1).numpy()
    
    return accuracy_score(y_train, y_pred)

def tune_hyperparameters(X_train, y_train, input_size, num_classes, n_trials=20):
    set_seed()
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, input_size, num_classes), n_trials=n_trials)
    return study.best_params
