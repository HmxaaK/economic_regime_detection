import streamlit as st
from back_testing.py import temporal_mapping
# ---------------- Streamlit UI ----------------
st.title('Economic Health Index')
st.subheader('Macroeconomic Indicators for Regime Classification')
st.write(
    "Economic regimes are characterized by dynamic complexity and the movement of variables. "
    "Some movements coincide with economic downturns or rallies, while others show fluctuations during transitions. "
    "To separate economic regimes at their peak and trough, the choice of these variables is crucial for accurate classification."
)

st.subheader('Coincident Indicators')
st.write(
    "The variables that show significant fluctuations during economic downturns and expansions are called coincident economic variables. "
    "The most significant of these in identifying economic regimes are Payroll employment, Civilian employment, Industrial production, and Unemployment rate."
)

st.subheader('Lagging Indicators')
st.write(
    "These indicators fluctuate after an economic downturn starts. "
    "The unemployment rate is the most significant indicator in this context, as it spikes greatly after the start of the recession and falls when the economy peaks. "
    "Other lagging variables include CPI and Bank interest rates (Hiroshi Iyetomi, 2020)."
)

st.subheader('Leading Indicators')
st.write(
    "Leading indicators include interest-rate spreads, credit and confidence gauges, and other series that typically fall before a cycle peak. "
    "For example, the Treasury yield curve (10-year minus 2-year or 3-month yields) has inverted before every post-1970 U.S. recession (Luca Benzoni, 2018). "
    "Likewise, broad money (M2) and commodity price indexes often reach their peak before an economic downturn (Michael D. Bordo, 2004). "
    "Consumer sentiment also tends to decline before the onset of a recession (Hiroshi Iyetomi, 2020). "
    "The Conference Board’s Leading Economic Index explicitly includes building permits and yield spreads as components."
)

st.subheader('Problem')
st.write(
    "The National Bureau of Economic Research (NBER) declares U.S. recessions months after they start. "
    "This lag makes real-time decision-making difficult for policymakers, investors, and businesses."
)

st.subheader('Solution')
st.write("We use historical monthly macroeconomic indicators and NBER labels to train a logistic regression model. This model can:")
st.write("- Estimate the probability of being in a recession this month.")
st.write("- Convert that probability into an Economic Health Score (0–10 scale).")

