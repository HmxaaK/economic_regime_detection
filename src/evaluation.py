import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


def model_performance(y_actual,y_predict):
  recall = recall_score(y_actual,y_predict)
  precision = precision_score(y_actual,y_predict)
  f1 = f1_score(y_actual,y_predict)
  cm = confusion_matrix(y_actual,y_predict)
  disp_cm = ConfusionMatrixDisplay(cm)
  print (f'Recall : {recall:.2f}')
  print (f'Precision : {precision:.2f}')
  print (f'F1 : {f1:.2f}')
  disp_cm.plot()
  plt.show()
  
def temporal_mapping(probs, labels):
    probs_recession = probs[:, 1]

    # --- Matplotlib plot ---
    fig1, ax = plt.subplots(figsize=(12, 6))
    ax.plot(labels.index, probs_recession, label="Probability of Recession")
    mask = (labels == 1).to_numpy()
    ax.fill_between(
        labels.index, 0, probs_recession,
        where=mask,
        color='grey', label="NBER = Recession"
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.legend()

    st.subheader("Recession Probability vs NBER Labels")
    st.pyplot(fig1)   # ✅ renders in Streamlit

    # --- Plotly plot ---
    economic_health_scores = np.round((probs[-39:, 0] * 10), 2)
    fig2 = px.line(
        x=labels.index[-39:], 
        y=economic_health_scores,
        labels={"x": "", "y": "Economic Health Score"},
        title="Economic Health Score (0–10)"
    )

    st.plotly_chart(fig2, use_container_width=True)   
