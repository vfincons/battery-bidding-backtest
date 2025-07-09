"""
Price Band Calculator for Battery Bidding Strategies - PRODUCTION VERSION
Diagnostics commented out - ready for exponential weighting implementation
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any
from scipy.interpolate import interp1d


class PriceBandCalculator:
    """
    Calculator for determining price bands based on historical RRP data using survival functions.

    Note: Diagnostic code commented out after identifying root cause of static price bands:
    - 96.4% data overlap in consecutive 28-day rolling windows
    - Extreme price events create "sticky" values in rolling window
    - Solution: Implement exponential weighting (to be added later)
    """

    def __init__(self):
        """Initialize with target dispatch probabilities."""
        self.target_dispatch_probabilities = {
            'PRICEBAND1': 100.0,    # Guaranteed dispatch (-$100)
            'PRICEBAND2': 85.0,     # 85% chance of dispatch
            'PRICEBAND3': 75.0,     # 75% chance of dispatch
            'PRICEBAND4': 65.0,     # 65% chance of dispatch
            'PRICEBAND5': 55.0,     # 55% chance of dispatch
            'PRICEBAND6': 45.0,     # 45% chance of dispatch
            'PRICEBAND7': 35.0,     # 35% chance of dispatch
            'PRICEBAND8': 25.0,     # 25% chance of dispatch
            'PRICEBAND9': 15.0,     # 15% chance of dispatch
            'PRICEBAND10': 2.5      # 2.5% chance of dispatch (rare high prices)
        }

        self.logger = logging.getLogger(__name__)

    def calculate_survival_function(self, peak_period_rrp_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate survival function - streamlined version."""
        clean_data = peak_period_rrp_data.dropna().values

        if len(clean_data) == 0:
            raise ValueError("No valid RRP data provided")

        # Calculate survival function using empirical approach
        sorted_data = np.sort(clean_data)
        n_total = len(sorted_data)

        # Create survival function using unique values with proper probabilities
        unique_prices = np.unique(sorted_data)
        survival_probabilities = []

        for price in unique_prices:
            # Count how many observations are >= this price
            count_ge = np.sum(sorted_data >= price)
            survival_prob = count_ge / n_total
            survival_probabilities.append(survival_prob)

        survival_probabilities = np.array(survival_probabilities)

        return unique_prices, survival_probabilities

    def interpolate_price_threshold(self, prices: np.ndarray, survival_probs: np.ndarray,
                                   target_probability: float) -> float:
        """Interpolate price threshold for target probability - streamlined version."""

        # Handle edge cases
        if target_probability >= survival_probs.max():
            return float(prices[survival_probs.argmax()])

        if target_probability <= survival_probs.min():
            return float(prices[survival_probs.argmin()])

        # Find bracketing points
        above_indices = np.where(survival_probs >= target_probability)[0]
        below_indices = np.where(survival_probs <= target_probability)[0]

        if len(above_indices) == 0 or len(below_indices) == 0:
            # Fallback to nearest neighbor
            nearest_idx = np.argmin(np.abs(survival_probs - target_probability))
            return float(prices[nearest_idx])

        # Get the closest bracketing points
        upper_bound_idx = above_indices[-1]  # Highest price with survival_prob >= target
        lower_bound_idx = below_indices[0]   # Lowest price with survival_prob <= target

        upper_price = prices[upper_bound_idx]
        upper_prob = survival_probs[upper_bound_idx]
        lower_price = prices[lower_bound_idx]
        lower_prob = survival_probs[lower_bound_idx]

        if abs(upper_prob - lower_prob) < 1e-10:
            # Probabilities are essentially identical
            price = (upper_price + lower_price) / 2
        else:
            # Linear interpolation
            if upper_prob != lower_prob:
                weight = (target_probability - lower_prob) / (upper_prob - lower_prob)
                price = lower_price + weight * (upper_price - lower_price)
            else:
                price = lower_price

        return float(price)

    def calculate_price_bands(self, peak_period_rrp_data: pd.Series) -> Dict[str, float]:
        """Calculate price bands - streamlined version."""

        if peak_period_rrp_data.empty:
            raise ValueError("Peak period RRP data is empty")

        # Calculate survival function
        prices, survival_probs = self.calculate_survival_function(peak_period_rrp_data)

        # Calculate price bands
        price_bands = {}

        for band_name, target_prob_percent in self.target_dispatch_probabilities.items():
            if band_name == 'PRICEBAND1':
                price_bands[band_name] = -100.0
            else:
                target_probability = target_prob_percent / 100.0
                price_threshold = self.interpolate_price_threshold(
                    prices, survival_probs, target_probability
                )
                price_bands[band_name] = price_threshold

        return price_bands

    def calculate_rolling_price_bands(self, data_manager, current_date: datetime) -> Dict[str, float]:
        """Calculate rolling price bands - streamlined version."""

        try:
            # Get historical data
            history_data = data_manager.get_history_window_data(current_date)

            if history_data.empty:
                raise ValueError(f"No historical data available for date {current_date.date()}")

            # Extract peak period RRP
            peak_period_rrp = history_data['RRP'].dropna()

            if len(peak_period_rrp) < 50:
                self.logger.warning(f"Limited data: only {len(peak_period_rrp)} observations")

            # Calculate price bands
            price_bands = self.calculate_price_bands(peak_period_rrp)

            return price_bands

        except Exception as e:
            self.logger.error(f"Failed to calculate rolling price bands: {e}")
            raise

    def validate_price_bands(self, price_bands: Dict[str, float]) -> bool:
        """Validate that price bands are in correct order."""
        band_values = []
        for i in range(1, 11):
            band_name = f'PRICEBAND{i}'
            if band_name in price_bands:
                band_values.append((i, price_bands[band_name]))

        for i in range(1, len(band_values)):
            current_band, current_value = band_values[i]
            prev_band, prev_value = band_values[i-1]

            if current_value < prev_value:
                self.logger.warning(
                    f"Price band validation failed: PRICEBAND{current_band} "
                    f"(${current_value:.2f}) < PRICEBAND{prev_band} (${prev_value:.2f}). "
                    f"Higher numbered bands should have higher thresholds."
                )
                return False
        return True

    def get_dispatch_probability_analysis(self, peak_period_rrp_data: pd.Series, price_bands: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Analyze actual dispatch probabilities for calculated price bands."""
        analysis = {}

        for band_name, threshold in price_bands.items():
            dispatch_events = (peak_period_rrp_data >= threshold).sum()
            total_periods = len(peak_period_rrp_data)
            actual_probability = dispatch_events / total_periods if total_periods > 0 else 0

            dispatched_rrp = peak_period_rrp_data[peak_period_rrp_data >= threshold]
            expected_rrp = dispatched_rrp.mean() if len(dispatched_rrp) > 0 else threshold

            target_probability = self.target_dispatch_probabilities.get(band_name, 0) / 100

            analysis[band_name] = {
                'threshold': threshold,
                'target_probability': target_probability,
                'actual_probability': actual_probability,
                'probability_error': actual_probability - target_probability,
                'dispatch_events': dispatch_events,
                'total_periods': total_periods,
                'expected_rrp_when_dispatched': expected_rrp,
                'expected_revenue_per_mw_per_interval': expected_rrp / 12
            }

        return analysis

    def export_price_band_analysis(self, peak_period_rrp_data: pd.Series, price_bands: Dict[str, float], output_path: str) -> pd.DataFrame:
        """Export detailed price band analysis to CSV."""
        analysis = self.get_dispatch_probability_analysis(peak_period_rrp_data, price_bands)

        analysis_df = pd.DataFrame.from_dict(analysis, orient='index')
        analysis_df.index.name = 'PriceBand'

        analysis_df['target_probability_percent'] = analysis_df['target_probability'] * 100
        analysis_df['actual_probability_percent'] = analysis_df['actual_probability'] * 100
        analysis_df['probability_error_percent'] = analysis_df['probability_error'] * 100

        analysis_df['band_number'] = analysis_df.index.str.extract(r'(\d+)').astype(int)
        analysis_df = analysis_df.sort_values('band_number')
        analysis_df = analysis_df.drop('band_number', axis=1)

        analysis_df.to_csv(output_path, float_format='%.4f')
        self.logger.info(f"Price band analysis exported to: {output_path}")

        return analysis_df

    def create_price_band_summary(self, price_bands: Dict[str, float]) -> str:
        """Create a formatted summary of price bands for logging."""
        summary_lines = ["Price Band Summary:"]

        for i in range(1, 11):
            band_name = f'PRICEBAND{i}'
            if band_name in price_bands:
                threshold = price_bands[band_name]
                target_prob = self.target_dispatch_probabilities.get(band_name, 0)
                summary_lines.append(
                    f"  {band_name}: ${threshold:8.2f} (target: {target_prob:5.1f}% dispatch prob.)"
                )

        return "\n".join(summary_lines)

    # TODO: Implement exponential weighting solution
    # def calculate_weighted_survival_function(self, data, dates):
    #     """
    #     Apply exponential decay weights: recent data gets higher weight
    #     Solution for sticky price bands due to 96.4% data overlap in rolling window
    #     """
    #     max_date = dates.max()
    #     weights = np.exp(-0.1 * (max_date - dates).dt.days)
    #     # Implementation to be added when ready for testing phase
    #     pass


# ============================================================================
# COMMENTED OUT DIAGNOSTIC CODE
# ============================================================================
# Root cause identified: 96.4% data overlap in consecutive 28-day windows
# Extreme price events create "sticky" values in rolling window
# Solution: Exponential weighting (to be implemented later)

"""
    # DIAGNOSTIC CODE - COMMENTED OUT FOR PRODUCTION

    def calculate_survival_function_with_diagnostics(self, peak_period_rrp_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        # Calculate survival function with detailed diagnostics.
        clean_data = peak_period_rrp_data.dropna().values

        if len(clean_data) == 0:
            raise ValueError("No valid RRP data provided")

        # Get basic statistics
        self.logger.info(f"=== SURVIVAL FUNCTION DIAGNOSTICS ===")
        self.logger.info(f"Total observations: {len(clean_data)}")
        self.logger.info(f"Price range: ${clean_data.min():.2f} to ${clean_data.max():.2f}")
        self.logger.info(f"Mean price: ${clean_data.mean():.2f}")
        self.logger.info(f"Std dev: ${clean_data.std():.2f}")

        # ... rest of diagnostic code ...

    def compare_consecutive_windows(self, data_manager, date1: datetime, date2: datetime) -> Dict[str, Any]:
        # Compare two consecutive rolling windows to see how much data actually changes.
        # ... diagnostic code ...

    def diagnose_rolling_window_data(self, data_manager, current_date: datetime) -> Dict[str, Any]:
        # Comprehensive diagnostic of rolling window data to identify static value causes.
        # ... diagnostic code ...

    def interpolate_with_diagnostics(self, prices: np.ndarray, survival_probs: np.ndarray,
                                     target_probability: float, band_name: str) -> float:
        # Interpolate with detailed diagnostics - CORRECTED VERSION.
        # ... diagnostic code ...

    def enhanced_price_band_diagnostic(self, peak_period_rrp_data: pd.Series) -> Dict[str, Any]:
        # Enhanced diagnostic specifically for price band calculation issues.
        # ... diagnostic code ...

    def calculate_rolling_price_bands_with_diagnostics(self, data_manager, current_date: datetime) -> Tuple[Dict[str, float], Dict[str, Any]]:
        # Calculate price bands with comprehensive diagnostics.
        # ... diagnostic code ...

    def _diagnose_price_band_issues(self, price_bands: Dict[str, float]) -> None:
        # Diagnose potential issues with calculated price bands.
        # ... diagnostic code ...
"""