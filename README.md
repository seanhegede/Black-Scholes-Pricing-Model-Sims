# Black-Scholes Option Pricing Model & Simulations

A Python-based implementation of the Black-Scholes pricing model with interactive simulations and a Streamlit web app for visualizing option pricing dynamics: https://black-scholes-pricing-model-sims-b4wyotm7sgbsfnmpnsomvb.streamlit.app/

---

##  Overview

This project implements the **Black-Scholes model** for pricing European call and put options, paired with Monte Carlo simulations to model underlying asset price paths. Includes both an exploratory Jupyter notebook and a deployable interactive app.

---

##  Features

- Black-Scholes closed-form pricing for **call and put options**
- Calculation of key **Greeks** (Delta, Gamma, Theta, Vega, Rho)
- **Monte Carlo simulations** of asset price paths using Geometric Brownian Motion
- Interactive **Streamlit dashboard** for real-time parameter adjustment
- Full **mathematical derivation** included as reference (`derivation.pdf`)

---

##  Repository Structure
```
├── black-scholes.ipynb     # Jupyter notebook: model walkthrough & simulations
├── black-scholes-app.py    # Streamlit app for interactive pricing
├── derivation.pdf          # Mathematical derivation of the Black-Scholes formula
├── requirements.txt        # Python dependencies
└── .devcontainer/          # Dev container configuration
```

---

##  Installation
```bash
git clone https://github.com/seanhegede/Black-Scholes-Pricing-Model-Sims.git
cd Black-Scholes-Pricing-Model-Sims
pip install -r requirements.txt
```

---

##  Usage

**Run the Streamlit app:**
```bash
streamlit run black-scholes-app.py
```

**Or explore the notebook:**
```bash
jupyter notebook black-scholes.ipynb
```

---

##  Model Inputs

| Parameter | Description |
|---|---|
| `S` | Current underlying asset price |
| `K` | Option strike price |
| `T` | Time to expiration (in years) |
| `r` | Risk-free interest rate |
| `σ` | Implied volatility of the underlying |

---

##  Tech Stack

- **Python** — core implementation
- **NumPy / SciPy** — mathematical computations
- **Matplotlib / Plotly** — visualizations
- **Streamlit** — interactive web app
- **Jupyter Notebook** — exploratory analysis

---

##  Reference

See `derivation.pdf` for the full stochastic calculus derivation of the Black-Scholes PDE and its closed-form solution.
