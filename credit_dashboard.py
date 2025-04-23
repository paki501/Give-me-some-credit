import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

st.set_page_config(layout="wide")

st.title("💳 Give Me Some Credit - Interactive Dashboard")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload the cs-training.csv file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, index_col=0)
    df.columns = ['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age', 
                  'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 
                  'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
                  'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 
                  'NumberOfDependents']

    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Sidebar - Feature filters
    st.sidebar.header("🔍 Data Exploration")
    if st.sidebar.checkbox("Show DataFrame"):
        st.dataframe(df.head())

    st.sidebar.markdown("Choose a visualization:")

    viz_option = st.sidebar.radio("Select Plot", ['Correlation Heatmap', 'Age Histogram', 'Income vs Age Scatter'])

    if viz_option == 'Correlation Heatmap':
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    elif viz_option == 'Age Histogram':
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['age'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    elif viz_option == 'Income vs Age Scatter':
        st.subheader("Monthly Income vs Age")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='age', y='MonthlyIncome', hue='SeriousDlqin2yrs', ax=ax)
        st.pyplot(fig)

    # Feature & target split
    X = df.drop("SeriousDlqin2yrs", axis=1)
    y = df["SeriousDlqin2yrs"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model selection
    st.sidebar.header("🧠 Choose Model")
    model_choice = st.sidebar.selectbox("Select Model", ['Random Forest', 'Gradient Boosting', 'XGBoost'])

    model_map = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    model = model_map[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Results
    st.subheader(f"📊 Classification Report - {model_choice}")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.subheader(f"🧮 Confusion Matrix - {model_choice}")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["No Default", "Default"], cmap='Blues', ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.subheader(f"📈 ROC Curve - {model_choice}")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
