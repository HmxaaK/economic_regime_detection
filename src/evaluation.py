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
    probs_recession = probs[:,1]
    plt.figure(figsize=(12, 6))
    plt.plot(labels.index, probs_recession, label="Probability of Recession")
    mask = (labels == 1).to_numpy()
    plt.fill_between(
        labels.index, 0, probs_recession,
        where=mask,
        color='grey', label="NBER = Recession"
    )
    plt.ylim(0,1)
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

    economic_health_scores = np.round((probs[-39:, 0] * 10),2)
    fig = px.line(x=labels.index[-39:],y=economic_health_scores,
                  labels={"x":"","y":"Economic Health Score"})
    fig.show()
