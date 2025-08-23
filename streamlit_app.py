import streamlit as st
st.title('Economic Health Index')
st.subheader('Macroeconomic Indicators for Regime Classification')
st.write('Economic regimes are characterized by dynamic complexity and the movement of variables. Some movements coincide with the economic downturns or rallies, while others show fluctuations during transitions. In order to separate economic regimes at their peak and trough, the choice of these variables is crucial for accurate classification') 
st.subheader('Coincident Indicators')
st.write('The variables that show significant fluctuations during the economic downturns and expansion are called coincident economic variables. The most significant of these in identifying economic regimes are Payroll employment, Civilian employment, Industrial production, and Unemployment rate.')
st.subheader('Lagging Indicators')
st.write('These indicators fluctuate after an economic downturn starts. The unemployment rate is the most significant indicator in this context, as it spikes greatly after the start of the recession and falls when the economic activity reaches its peak. Significant lagging variables include CPI and Bank interest rates, as indicated by (Hiroshi Iyetomi, 2020).')
