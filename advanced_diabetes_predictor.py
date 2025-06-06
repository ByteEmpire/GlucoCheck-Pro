import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

# ---- 1. Load & Preprocess Data ----
def load_data():
    df = pd.read_csv("diabetes.csv")
    
    # Feature engineering (add BMI categories)
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Separate features and target
    X = df.drop(["Outcome", "BMI_Category"], axis=1)
    y = df["Outcome"]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, scaler, df

# ---- 2. Hyperparameter Tuning (XGBoost) ----
def train_xgboost(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.7, 0.9]
    }
    
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    print("Best params:", grid_search.best_params_)
    return grid_search.best_estimator_

# ---- 3. Evaluate Model ----
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix (Plotly)
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), 
                    x=['No Diabetes', 'Diabetes'], y=['No Diabetes', 'Diabetes'])
    fig.update_layout(title="Confusion Matrix")
    fig.write_html("confusion_matrix.html")
    
    # Feature Importance (Plotly)
    feat_imp = pd.DataFrame({
        'Feature': ['Pregnancies', 'Glucose', 'BP', 'Skin Thickness', 'Insulin', 'BMI', 'Pedigree', 'Age'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title='Feature Importance')
    fig.write_html("feature_importance.html")

# ---- 4. Save Artifacts ----
def save_artifacts(model, scaler, df):
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    df.to_csv("enhanced_diabetes.csv", index=False)
    print("Artifacts saved!")

# ---- Main ----
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, df = load_data()
    model = train_xgboost(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_artifacts(model, scaler, df)