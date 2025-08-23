# src/evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def model_performance(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    st.subheader("Model Performance: Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

def temporal_mapping(probs, labels):
    # --- Matplotlib plot ---
    probs_recession = probs[:, 1]
    fig1, ax = plt.subplots(figsize=(12, 6))
    ax.plot(labels.index, probs_recession, label="Probability of Recession")
    mask = (labels == 1).to_numpy()
    ax.fill_between(labels.index, 0, probs_recession,
                    where=mask, color='grey', label="NBER = Recession")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.legend()
    st.subheader("Recession Probability vs NBER Labels")
    st.pyplot(fig1)
    plt.close(fig1)

    # --- Plotly plot ---
    economic_health_scores = np.round((probs[-39:, 0] * 10), 2)
    fig2 = px.line(
        x=labels.index[-39:], 
        y=economic_health_scores,
        labels={"x": "", "y": "Economic Health Score"},
        title="Economic Health Score (0–10)"
    )
    st.plotly_chart(fig2, use_container_width=True)
