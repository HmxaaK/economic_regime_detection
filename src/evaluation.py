# src/evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
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
