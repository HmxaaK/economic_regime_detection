def temporal_mapping(probs, labels, window=39):
    # --- 1. Matplotlib plot ---
    probs_recession = probs[:, 1]
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

    # --- 2. Plotly plot ---
    economic_health_scores = np.round((probs[-window:, 0] * 10), 2)
    fig2 = px.line(
        x=labels.index[-window:],
        y=economic_health_scores,
        labels={"x": "", "y": "Economic Health Score"}
    )

    return fig1, fig2
