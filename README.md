# Economic Regime Detection & Nowcasting

This project uses **Logistic Regression** on macroeconomic indicators to nowcast the current economic regime (recession or expansion) and compute an **Economic Health Score**.

---

## 📌 Problem
The National Bureau of Economic Research (NBER) declares U.S. recessions **months after they start**. This lag makes real-time decision-making difficult for policymakers, investors, and businesses.

## 💡 Solution
We use historical monthly macroeconomic indicators and NBER labels to train a logistic regression model. This model can:
- Estimate the probability of being in a recession *this month*.
- Convert that probability into an **Economic Health Score** (0–10 scale).

---

## 🛠 Features
- Feature engineering with polynomial terms, logs, and differences.
- Logistic regression with time-series cross-validation and hyperparameter tuning.
- Probability-to-score transformation for intuitive interpretation.
- Historical backtesting with visualizations.

---

## 📂 Project Structure
- `data/` → CSV datasets  
- `src/` → ML pipeline code  
- `app/` → Streamlit app (future)  
- `requirements.txt` → Python dependencies  
- `README.md` → Documentation  

---

## 🚀 Usage

1. **Clone the repository**
```bash
git clone https://github.com/HmxaaK/economic_regime_detection.git
cd economic_regime_detection```




