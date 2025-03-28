from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred)) 
    return accuracy, classification_report(y_test, y_pred), confusion_matrix(y_test, y_pred)


def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])  # features
    y = df[target_column]  # target variable

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    return X_scaled, y

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=100,  # num of boosting stages
        learning_rate=0.1,  # contribution of each tree
        max_depth=3,  # limits complexity of trees
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model