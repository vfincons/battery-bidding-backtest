"""
Configuration module for Battery Bidding Backtest System
"""
from datetime import datetime, time
from typing import Dict, Any, Tuple
import os


class Configuration:
    """
    Configuration class to store all parameters for the battery bidding backtest.
    """

    def __init__(self):
        """Initialize configuration with default values."""
        # Date parameters
        self.start_date: str = "2024/03/01 00:00:00"
        self.end_date: str = "2024/08/31 23:55:00"
        self.analysis_start_date: str = "2024/04/01 00:00:00"  # Allow for 28-day history

        # Peak period definition
        self.peak_period_start_time: time = time(17, 0)  # 17:00
        self.peak_period_end_time: time = time(19, 30)  # 19:30

        # Battery parameters
        self.battery_parameters: Dict[str, Any] = {
            'max_discharge_power': 100,  # MW
            'energy_capacity': 400,  # MWh (optional)
            'ramp_limits': (48, 192),  # MW/min (min, max)
            'TLF': 0.9  # Transmission Loss Factor
        }

        # Data parameters
        self.region_id: str = "NSW1"
        self.history_window_days: int = 28

        # File paths - Updated to use specified directories
        self.raw_data_cache: str = r"C:\Users\Anderson\Machine Learning\NEM\NEMOSIS_raw_data_cache"
        self.output_directory: str = r"C:\Users\Anderson\Machine Learning\NEM\NEMOSIS_output_data\Battery_Bidding"

        # Nemosis parameters
        self.nemosis_table: str = "DISPATCHPRICE"
        self.nemosis_columns: list = ['REGIONID', 'SETTLEMENTDATE', 'RRP', 'INTERVENTION']

    def validate_configuration(self) -> bool:
        """
        Validate all configuration parameters.

        Returns:
        --------
        bool
            True if configuration is valid, False otherwise

        Raises:
        -------
        ValueError
            If any configuration parameter is invalid
        """
        # Validate date format and order
        try:
            start_dt = datetime.strptime(self.start_date, "%Y/%m/%d %H:%M:%S")
            end_dt = datetime.strptime(self.end_date, "%Y/%m/%d %H:%M:%S")
            analysis_start_dt = datetime.strptime(self.analysis_start_date, "%Y/%m/%d %H:%M:%S")

            if start_dt >= end_dt:
                raise ValueError("start_date must be before end_date")

            if analysis_start_dt < start_dt:
                raise ValueError("analysis_start_date must be after start_date")

            if analysis_start_dt >= end_dt:
                raise ValueError("analysis_start_date must be before end_date")

        except ValueError as e:
            if "time data" in str(e):
                raise ValueError("Date format must be 'YYYY/MM/DD HH:MM:SS'")
            raise

        # Validate peak period times
        if self.peak_period_start_time >= self.peak_period_end_time:
            raise ValueError("peak_period_start_time must be before peak_period_end_time")

        # Validate battery parameters
        if self.battery_parameters['max_discharge_power'] <= 0:
            raise ValueError("max_discharge_power must be positive")

        if self.battery_parameters['energy_capacity'] <= 0:
            raise ValueError("energy_capacity must be positive")

        if (self.battery_parameters['ramp_limits'][0] < 0 or
                self.battery_parameters['ramp_limits'][1] < 0):
            raise ValueError("ramp_limits must be positive")

        if (self.battery_parameters['ramp_limits'][0] >
                self.battery_parameters['ramp_limits'][1]):
            raise ValueError("ramp_limits[0] must be <= ramp_limits[1]")

        if not 0 < self.battery_parameters['TLF'] <= 1:
            raise ValueError("TLF must be between 0 and 1")

        # Validate region ID
        valid_regions = ['NSW1', 'QLD1', 'SA1', 'TAS1', 'VIC1']
        if self.region_id not in valid_regions:
            raise ValueError(f"region_id must be one of {valid_regions}")

        # Validate history window
        if self.history_window_days <= 0:
            raise ValueError("history_window_days must be positive")

        # Create directories if they don't exist
        try:
            os.makedirs(self.raw_data_cache, exist_ok=True)
            os.makedirs(self.output_directory, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Failed to create directories: {str(e)}")

        return True

    def get_battery_max_ramp_5min(self) -> float:
        """
        Get maximum ramp rate for 5-minute intervals.

        Returns:
        --------
        float
            Maximum ramp rate in MW for 5-minute interval
        """
        return self.battery_parameters['ramp_limits'][1] * (5 / 60)  # Convert MW/min to MW/5min

    def get_battery_min_ramp_5min(self) -> float:
        """
        Get minimum ramp rate for 5-minute intervals.

        Returns:
        --------
        float
            Minimum ramp rate in MW for 5-minute interval
        """
        return self.battery_parameters['ramp_limits'][0] * (5 / 60)  # Convert MW/min to MW/5min

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.

        Returns:
        --------
        Dict[str, Any]
            Configuration as dictionary
        """
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'analysis_start_date': self.analysis_start_date,
            'peak_period_start_time': self.peak_period_start_time.strftime("%H:%M"),
            'peak_period_end_time': self.peak_period_end_time.strftime("%H:%M"),
            'battery_parameters': self.battery_parameters,
            'region_id': self.region_id,
            'history_window_days': self.history_window_days,
            'raw_data_cache': self.raw_data_cache,
            'output_directory': self.output_directory
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""Battery Backtest Configuration:
        Start Date: {self.start_date}
        End Date: {self.end_date}
        Analysis Start Date: {self.analysis_start_date}
        Peak Period: {self.peak_period_start_time} - {self.peak_period_end_time}
        Region: {self.region_id}
        Battery Power: {self.battery_parameters['max_discharge_power']} MW
        Battery Energy: {self.battery_parameters['energy_capacity']} MWh
        Ramp Limits: {self.battery_parameters['ramp_limits']} MW/min
        TLF: {self.battery_parameters['TLF']}
        Raw Data Cache: {self.raw_data_cache}
        Output Directory: {self.output_directory}
        """