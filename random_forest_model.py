from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_random_forest(
        
    X_train, y_train, 
    n_estimators=200,        # More trees for better generalization
    max_depth=15,            # Limit tree depth
    min_samples_split=10,    # Prevents small, overfitting splits
    min_samples_leaf=5,      # Prevents overly specific rules
    random_state=42
):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced' # Helps with class imbalance
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # testing accuracy
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Testing Accuracy: {test_accuracy}")
    print("Classification Report (Test Data):")
    print(classification_report(y_test, y_test_pred))  
    print("Confusion Matrix (Test Data):")
    print(confusion_matrix(y_test, y_test_pred)) 
    
    return train_accuracy, test_accuracy, classification_report(y_test, y_test_pred), confusion_matrix(y_test, y_test_pred)


def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])  # features
    y = df[target_column]  # target variable

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    return X_scaled, y
