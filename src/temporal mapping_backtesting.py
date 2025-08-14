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
