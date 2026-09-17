# Sovereign Debt Sustainability Analysis (DSA)

This repository contains an Object-Oriented Python pipeline designed for macroeconomic data engineering and stochastic modeling, mirroring the rigorous standards of sovereign advisory and central banking.

### Architecture & Features
* **Object-Oriented Design:** Encapsulated logic within the `SovereignDebtAnalyzer` class.
* **Robust Data Cleaning:** Handles missing values via cubic interpolation and neutralizes statistical outliers using Interquartile Range (IQR) capping.
* **Stochastic Modeling:** Implements a Monte Carlo simulation (2,000 iterations) to project Debt-to-GDP trajectories under macroeconomic uncertainty (shocks on growth and interest rates).
* **Institutional Data Viz:** Generates high-resolution "Fan Charts" (confidence intervals) typical of central bank and IMF publications, using `seaborn` and `matplotlib`.

### Execution
The script generates a synthetic dataset upon execution for immediate testing.
