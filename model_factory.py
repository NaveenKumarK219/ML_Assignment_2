import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_data():
    """
    Loads the Breast Cancer dataset, splits it, and fits a scaler.
    Returns training data, test data, and the fitted scaler.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    # Stratify ensures the class distribution is maintained
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X_train)

    return X_train, X_test, y_train, y_test, scaler

def get_model(model_name, X_train, y_train, scaler):
    """
    Trains and returns the requested model.
    Applies scaling within this function if the model requires it.
    """
    # 1. Prepare data (Scale if necessary for specific models)
    if model_name in ["Logistic Regression", "KNN"]:
        X_train_processed = scaler.transform(X_train)
    else:
        X_train_processed = X_train # Trees/NB handle raw data well

    # 2. Initialize Model
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 3. Train
    model.fit(X_train_processed, y_train)

    return model