import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple

# ==========================================
# CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SovereignDebtAnalyzer:
    """
    Object-Oriented pipeline for Macroeconomic Data Processing 
    and Stochastic Debt Sustainability Analysis (DSA).
    """
    
    def __init__(self, target_country: str, start_year: int = 2010, end_year: int = 2023):
        self.target_country = target_country
        self.start_year = start_year
        self.end_year = end_year
        self.raw_data = pd.DataFrame()
        self.clean_data = pd.DataFrame()
        
    def ingest_synthetic_panel(self) -> None:
        """Generates a complex synthetic macro-financial panel dataset."""
        logging.info(f"Ingesting macro panel data from {self.start_year} to {self.end_year}...")
        np.random.seed(42)
        years = list(range(self.start_year, self.end_year + 1))
        n_years = len(years)
        
        # Simulating correlated macroeconomic variables
        gdp_growth = np.random.normal(2.0, 1.5, n_years)
        # Primary balance negatively correlated with GDP shocks
        primary_balance = gdp_growth * 0.5 + np.random.normal(-1.0, 1.0, n_years)
        
        self.raw_data = pd.DataFrame({
            'Year': years,
            'Country': [self.target_country] * n_years,
            'GDP_Growth_Pct': gdp_growth,
            'Primary_Balance_GDP_Pct': primary_balance,
            'Effective_Interest_Rate_Pct': np.random.uniform(1.5, 4.5, n_years)
        })
        
        # Inject artificial anomalies to demonstrate robust cleaning
        self.raw_data.loc[3, 'GDP_Growth_Pct'] = np.nan
        self.raw_data.loc[10, 'Primary_Balance_GDP_Pct'] = -99.9 # Outlier
        logging.info(f"Dataset ingested with shape: {self.raw_data.shape}")

    def clean_pipeline(self) -> None:
        """Cleans data using interpolation and IQR-based outlier detection."""
        logging.info("Executing data cleaning pipeline...")
        df = self.raw_data.copy()
        
        # 1. Handle missing values via cubic interpolation
        df['GDP_Growth_Pct'] = df['GDP_Growth_Pct'].interpolate(method='cubic')
        
        # 2. Outlier detection and capping using Interquartile Range (IQR)
        q1 = df['Primary_Balance_GDP_Pct'].quantile(0.25)
        q3 = df['Primary_Balance_GDP_Pct'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        df['Primary_Balance_GDP_Pct'] = np.where(
            df['Primary_Balance_GDP_Pct'] < lower_bound, lower_bound,
            np.where(df['Primary_Balance_GDP_Pct'] > upper_bound, upper_bound, df['Primary_Balance_GDP_Pct'])
        )
        
        self.clean_data = df
        logging.info("Data cleaning completed successfully.")

    def monte_carlo_dsa(self, current_debt_ratio: float, simulations: int = 1000, horizon: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a stochastic Debt Sustainability Analysis (DSA) using Monte Carlo simulations.
        Formula: D(t) = D(t-1) * (1 + r - g) / (1 + g) - PB
        """
        logging.info(f"Running Monte Carlo DSA ({simulations} simulations for {horizon} years)...")
        
        mu_g = self.clean_data['GDP_Growth_Pct'].mean() / 100
        std_g = self.clean_data['GDP_Growth_Pct'].std() / 100
        mu_r = self.clean_data['Effective_Interest_Rate_Pct'].mean() / 100
        mu_pb = self.clean_data['Primary_Balance_GDP_Pct'].mean() / 100
        
        debt_trajectories = np.zeros((simulations, horizon + 1))
        debt_trajectories[:, 0] = current_debt_ratio
        
        for t in range(1, horizon + 1):
            g_shock = np.random.normal(mu_g, std_g, simulations)
            # Debt accumulation dynamic
            snowball_effect = (1 + mu_r - g_shock) / (1 + g_shock)
            debt_trajectories[:, t] = debt_trajectories[:, t-1] * snowball_effect - mu_pb
            
        return debt_trajectories, np.percentile(debt_trajectories, [10, 50, 90], axis=0)

    def generate_dashboard(self, trajectories: np.ndarray, percentiles: np.ndarray, horizon: int) -> None:
        """Generates an institutional-grade visualization using Seaborn & Matplotlib."""
        logging.info("Generating macroeconomic dashboard...")
        sns.set_theme(style="whitegrid", palette="muted")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Historical Macroeconomic Fundamentals
        ax1 = axes[0]
        ax1.plot(self.clean_data['Year'], self.clean_data['GDP_Growth_Pct'], label='GDP Growth (%)', marker='o', linewidth=2)
        ax1.plot(self.clean_data['Year'], self.clean_data['Primary_Balance_GDP_Pct'], label='Primary Balance (%)', linestyle='--', linewidth=2)
        ax1.set_title(f'Historical Macro Fundamentals - {self.target_country}', fontsize=14, pad=15)
        ax1.set_ylabel('Percentage (%)')
        ax1.legend()
        
        # Plot 2: Monte Carlo Debt Projections (Fan Chart)
        ax2 = axes[1]
        projection_years = np.arange(self.end_year, self.end_year + horizon + 1)
        
        # Plot a sample of individual stochastic trajectories
        for i in range(100):
            ax2.plot(projection_years, trajectories[i, :] * 100, color='gray', alpha=0.05)
            
        # Plot Confidence Intervals (10th, 50th, 90th percentiles)
        ax2.plot(projection_years, percentiles[1] * 100, color='darkred', linewidth=3, label='Median Projection')
        ax2.fill_between(projection_years, percentiles[0] * 100, percentiles[2] * 100, color='red', alpha=0.2, label='80% Confidence Interval')
        
        ax2.set_title(f'Stochastic Debt Sustainability Analysis (DSA)', fontsize=14, pad=15)
        ax2.set_ylabel('Debt-to-GDP Ratio (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('dsa_dashboard.png', dpi=300)
        logging.info("Dashboard saved as 'dsa_dashboard.png'. Process complete.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Initialize the analyzer for a hypothetical sovereign state
    analyzer = SovereignDebtAnalyzer(target_country="Republic of Alpha", start_year=2010, end_year=2025)
    
    # Run the pipeline
    analyzer.ingest_synthetic_panel()
    analyzer.clean_pipeline()
    
    # Run Monte Carlo DSA (Starting at 85% Debt-to-GDP)
    trajectories, percentiles = analyzer.monte_carlo_dsa(current_debt_ratio=0.85, simulations=2000, horizon=5)
    
    # Visualize and export
    analyzer.generate_dashboard(trajectories, percentiles, horizon=5)
