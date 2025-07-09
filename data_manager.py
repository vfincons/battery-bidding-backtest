"""
Data management module for Battery Bidding Backtest System
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Optional, Dict, Any
import logging
import nemosis
from config import Configuration


class DataManager:
    """
    Data management class for handling AEMO DISPATCHPRICE data using NEMOSIS.
    """

    def __init__(self, configuration: Configuration):
        """
        Initialize DataManager with configuration.

        Parameters:
        -----------
        configuration : Configuration
            Configuration object containing all parameters
        """
        self.config = configuration
        self.rrp_data: Optional[pd.DataFrame] = None
        self.logger = logging.getLogger(__name__)

    def download_dispatch_price(self, start_time: str, end_time: str) -> pd.DataFrame:
        """
        Download DISPATCHPRICE for the specified region using NEMOSIS.

        Parameters:
        -----------
        start_time : str
            Start time for data download (format: 'YYYY/MM/DD HH:MM:SS')
        end_time : str
            End time for data download (format: 'YYYY/MM/DD HH:MM:SS')

        Returns:
        --------
        pd.DataFrame
            Filtered and processed DISPATCHPRICE data with INTERVENTION==0
        """
        try:
            self.logger.info(f"Downloading DISPATCHPRICE data from {start_time} to {end_time}")

            # Download DISPATCH_PRICE filtering for specified REGIONID and only non-intervention runs
            dispatch_price = nemosis.dynamic_data_compiler(
                start_time,
                end_time,
                self.config.nemosis_table,
                self.config.raw_data_cache,
                filter_cols=['REGIONID', 'INTERVENTION'],
                filter_values=([self.config.region_id], [0]),
                select_columns=self.config.nemosis_columns,
                fformat='parquet',
                keep_csv=False
            )

            # Remove the INTERVENTION column after filtering
            dispatch_price = dispatch_price.drop(columns=['INTERVENTION'])

            # Ensure SETTLEMENTDATE is datetime
            dispatch_price['SETTLEMENTDATE'] = pd.to_datetime(dispatch_price['SETTLEMENTDATE'])

            # Sort by settlement date
            dispatch_price = dispatch_price.sort_values('SETTLEMENTDATE').reset_index(drop=True)

            self.logger.info(f"Downloaded {len(dispatch_price)} records")

            return dispatch_price

        except Exception as e:
            self.logger.error(f"Error downloading DISPATCHPRICE data: {str(e)}")
            raise

    def load_rrp_data(self) -> None:
        """
        Load RRP data for the full date range specified in configuration.

        Raises:
        -------
        Exception
            If data loading fails
        """
        try:
            self.logger.info("Loading RRP data for full date range")

            # Download data for full range
            self.rrp_data = self.download_dispatch_price(
                self.config.start_date,
                self.config.end_date
            )

            # Validate data integrity
            self._validate_data_integrity()

            self.logger.info(f"Successfully loaded {len(self.rrp_data)} RRP records")

        except Exception as e:
            self.logger.error(f"Failed to load RRP data: {str(e)}")
            raise

    def _validate_data_integrity(self) -> None:
        """
        Validate the integrity of loaded RRP data.

        Raises:
        -------
        ValueError
            If data integrity checks fail
        """
        if self.rrp_data is None or len(self.rrp_data) == 0:
            raise ValueError("No RRP data loaded")

        # Check for required columns
        required_columns = ['REGIONID', 'SETTLEMENTDATE', 'RRP']
        missing_columns = [col for col in required_columns if col not in self.rrp_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for null values in critical columns
        null_counts = self.rrp_data[required_columns].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values found in critical columns: {null_counts[null_counts > 0]}")

        # Check region ID consistency
        unique_regions = self.rrp_data['REGIONID'].unique()
        if len(unique_regions) != 1 or unique_regions[0] != self.config.region_id:
            raise ValueError(f"Expected only {self.config.region_id}, found: {unique_regions}")

        # Check for 5-minute intervals
        self.rrp_data['time_diff'] = self.rrp_data['SETTLEMENTDATE'].diff()
        expected_interval = pd.Timedelta(minutes=5)
        non_standard_intervals = self.rrp_data[
            (self.rrp_data['time_diff'] != expected_interval) &
            (self.rrp_data['time_diff'].notna())
            ]

        if len(non_standard_intervals) > 0:
            self.logger.warning(f"Found {len(non_standard_intervals)} non-standard intervals")

        # Remove the temporary column
        self.rrp_data = self.rrp_data.drop(columns=['time_diff'])

        self.logger.info("Data integrity validation passed")

    def get_history_window_data(self, current_date: datetime) -> pd.DataFrame:
        """
        Get 28-day history window data ending on the previous day, filtered for the peak period.

        Parameters:
        -----------
        current_date : datetime
            Current date for which to get a history window

        Returns:
        --------
        pd.DataFrame
            Peak period RRP data for the 28-day history window
        """
        if self.rrp_data is None:
            raise ValueError("RRP data not loaded. Call load_rrp_data() first.")

        # Calculate history window dates
        end_date = current_date - timedelta(days=1)
        start_date = end_date - timedelta(days=self.config.history_window_days - 1)

        self.logger.debug(f"Getting history window from {start_date.date()} to {end_date.date()}")

        # Filter by date range
        history_data = self.rrp_data[
            (self.rrp_data['SETTLEMENTDATE'].dt.date >= start_date.date()) &
            (self.rrp_data['SETTLEMENTDATE'].dt.date <= end_date.date())
            ].copy()

        # Filter by peak period times
        peak_period_data = self._filter_by_peak_period(history_data)

        self.logger.debug(f"Found {len(peak_period_data)} peak period records in history window")

        return peak_period_data

    def get_current_day_data(self, current_date: datetime) -> pd.DataFrame:
        """
        Get RRP data for the current day (all 5-minute intervals).

        Parameters:
        -----------
        current_date : datetime
            Date for which to get RRP data

        Returns:
        --------
        pd.DataFrame
            RRP data for the current day
        """
        if self.rrp_data is None:
            raise ValueError("RRP data not loaded. Call load_rrp_data() first.")

        # Filter by current date
        current_day_data = self.rrp_data[
            self.rrp_data['SETTLEMENTDATE'].dt.date == current_date.date()
            ].copy()

        self.logger.debug(f"Found {len(current_day_data)} records for {current_date.date()}")

        return current_day_data

    def _filter_by_peak_period(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data by peak period times.

        Parameters:
        -----------
        data : pd.DataFrame
            Data to filter

        Returns:
        --------
        pd.DataFrame
            Data filtered to peak period times
        """
        if data.empty:
            return data
        # Extract time from SETTLEMENTDATE
        data_with_time = data.copy()
        data_with_time['time'] = data_with_time['SETTLEMENTDATE'].dt.time

        # Filter by peak period
        peak_period_data = data_with_time[
            (data_with_time['time'] >= self.config.peak_period_start_time) &
            (data_with_time['time'] <= self.config.peak_period_end_time)
            ].copy()

        # Remove temporary time column
        peak_period_data = peak_period_data.drop(columns=['time'])

        return peak_period_data

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded RRP data.

        Returns:
        --------
        Dict[str, Any]
            Summary statistics
        """
        if self.rrp_data is None:
            return {"status": "No data loaded"}

        summary = {
            "total_records": len(self.rrp_data),
            "date_range": {
                "start": self.rrp_data['SETTLEMENTDATE'].min(),
                "end": self.rrp_data['SETTLEMENTDATE'].max()
            },
            "rrp_statistics": {
                "mean": self.rrp_data['RRP'].mean(),
                "median": self.rrp_data['RRP'].median(),
                "std": self.rrp_data['RRP'].std(),
                "min": self.rrp_data['RRP'].min(),
                "max": self.rrp_data['RRP'].max(),
                "q25": self.rrp_data['RRP'].quantile(0.25),
                "q75": self.rrp_data['RRP'].quantile(0.75)
            },
            "region": self.config.region_id,
            "unique_dates": self.rrp_data['SETTLEMENTDATE'].dt.date.nunique()
        }

        return summary

    def export_data_sample(self, n_records: int = 100) -> pd.DataFrame:
        """
        Export a sample of the loaded data for inspection.

        Parameters:
        -----------
        n_records : int, optional
            Number of records to sample (default: 100)

        Returns:
        --------
        pd.DataFrame
            Sample of the data
        """
        if self.rrp_data is None:
            raise ValueError("RRP data not loaded. Call load_rrp_data() first.")

        return self.rrp_data.head(n_records)

    def check_data_availability(self, date: datetime) -> bool:
        """
        Check if data is available for a specific date.

        Parameters:
        -----------
        date : datetime
            Date to check

        Returns:
        --------
        bool
            True if data is available, False otherwise
        """
        if self.rrp_data is None:
            return False

        return date.date() in self.rrp_data['SETTLEMENTDATE'].dt.date.values