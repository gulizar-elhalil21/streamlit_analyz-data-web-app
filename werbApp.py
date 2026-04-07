from pyexpat import model

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay



def main():
    st.title("Binary Calassification Mashrooms")
    st.sidebar.title("Binary calssification model")
    st.markdown("are your mashrroms edible or poisonous?🍄")
    st.sidebar.markdown("this is a bianary calssification model that predicts whether a mashroom is edible or poisonous based on its features.")

    def data():
        data = pd.read_csv("mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    @st.cache_data
    def split(df):
        y = df['class']
        x = df.drop(columns=["class"])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
        return x_train, x_test, y_train, y_test

    def plot_metrics(model, x_test, y_test, metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
            st.pyplot(disp.figure_)
        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            disp = RocCurveDisplay.from_estimator(model, x_test, y_test)
            disp.plot()
            plt.plot([0,1], [0,1], 'k--')
            plt.title('ROC Curve - Replaces deprecated plot_roc_curve')
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot()






    df = data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ["edible", "poisonous"]
    st.sidebar.subheader("choose classifier")
    classifier = st.sidebar.selectbox("classifier", ("Support Vector Machine (svm)", "Logistic Regression", "Random Forest"))
    if classifier == "Support Vector Machine (svm)":
        st.sidebar.subheader("model hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
        kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")
        metrics = st.sidebar.multiselect("choose metrics to plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("classify", key="classify"):
            st.subheader("Support Vector Machine (svm) results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("precision: ", precision_score(y_test, y_pred, labels=class_names))
            st.write("recall: ", recall_score(y_test, y_pred, labels=class_names))
            plot_metrics(model, x_test, y_test, metrics)
    elif classifier == "Logistic Regression":
        st.sidebar.subheader("model hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")
        metrics = st.sidebar.multiselect("choose metrics to plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("classify", key="classify_LR"):
            st.subheader("Logistic Regression results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("precision: ", precision_score(y_test, y_pred, labels=class_names))
            st.write("recall: ", recall_score(y_test, y_pred, labels=class_names))
            plot_metrics(model, x_test, y_test, metrics)
    elif classifier == "Random Forest":
        st.sidebar.subheader("model hyperparameters")
        n_estimators = st.sidebar.number_input("number of trees in the forest", 100, 500, step=10, key="n_estimators")
        max_depth = st.sidebar.number_input("maximum depth of the tree", 1, 20, step=1, key="max_depth")
        bootstrap = st.sidebar.radio("bootstrap samples when building trees", ("True", "False"), key="bootstrap")
        metrics = st.sidebar.multiselect("choose metrics to plot", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
        if st.sidebar.button("classify", key="classify_RF"):
            st.subheader("Random Forest results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=(bootstrap == "True"), n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy)
            st.write("precision: ", precision_score(y_test, y_pred, labels=class_names))
            st.write("recall: ", recall_score(y_test, y_pred, labels=class_names))
            plot_metrics(model, x_test, y_test, metrics)



    if st.sidebar.checkbox("show raw data"):
        st.subheader("mashrooms data")
        st.dataframe(df)


if __name__ == "__main__":
    main()
