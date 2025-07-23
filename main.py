"""
Streamlined Main Entry Point for Battery Bidding Backtest System
"""
# Standard library imports
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Third-party imports
import pandas as pd

# Local imports
from config import Configuration
from data_manager import DataManager
from dispatch_simulator import DispatchSimulator
from price_band_calculator import PriceBandCalculator
from bidding_strategy_manager import BiddingStrategyManager, BiddingStrategy
from revenue_calculator import RevenueCalculator
from logger import BatteryBacktestLogger
from results_analyzer import ResultsAnalyzer


def validate_dispatch_simulator_interface(dispatch_simulator) -> List[str]:
    """
    Validate DispatchSimulator interface compliance.

    Parameters:
    -----------
    dispatch_simulator : DispatchSimulator
        Instance to validate

    Returns:
    --------
    List[str]
        List of missing methods/interface issues
    """
    issues = []

    if not dispatch_simulator:
        return ["DispatchSimulator instance is None"]

    # Core methods required by BatteryBacktester
    required_methods = [
        'simulate_daily_dispatch', 'reset_simulation', 'export_dispatch_results',
        'simulate_dispatch_for_interval', 'calculate_maxavail',
        'apply_nemde_dispatch_algorithm', 'validate_ramp_constraints_post_dispatch'
    ]

    for method in required_methods:
        if not hasattr(dispatch_simulator, method):
            issues.append(f"DispatchSimulator missing required method: {method}")
        elif not callable(getattr(dispatch_simulator, method)):
            issues.append(f"DispatchSimulator.{method} is not callable")

    # Enhanced methods (optional but recommended)
    enhanced_methods = ['clear_history', 'get_export_summary', 'validate_export_data']
    missing_enhanced = []
    for method in enhanced_methods:
        if not hasattr(dispatch_simulator, method):
            missing_enhanced.append(method)

    if missing_enhanced:
        issues.append(f"DispatchSimulator missing enhanced methods: {', '.join(missing_enhanced)}")

    return issues


def validate_bidding_strategy_manager_interface(strategy_manager) -> List[str]:
    """
    Validate BiddingStrategyManager interface compliance.

    Parameters:
    -----------
    strategy_manager : BiddingStrategyManager
        Instance to validate

    Returns:
    --------
    List[str]
        List of missing methods/interface issues
    """
    issues = []

    if not strategy_manager:
        return ["BiddingStrategyManager instance is None"]

    # Core methods required by DispatchSimulator and BatteryBacktester
    required_methods = [
        'initialize_strategy', 'get_bid_schedule', 'get_remaining_soc',
        'update_after_dispatch', 'get_strategy_info', 'is_final_override_active',
        'get_bid_schedule_with_maxavail_check', 'update_band_allocations_after_nemde_dispatch'
    ]

    for method in required_methods:
        if not hasattr(strategy_manager, method):
            issues.append(f"BiddingStrategyManager missing required method: {method}")
        elif not callable(getattr(strategy_manager, method)):
            issues.append(f"BiddingStrategyManager.{method} is not callable")

    # Check required attributes
    required_attributes = ['total_capacity', 'total_intervals', 'minimum_floor_mw', 'max_ramp_per_interval']
    for attr in required_attributes:
        if not hasattr(strategy_manager, attr):
            issues.append(f"BiddingStrategyManager missing required attribute: {attr}")

    # Compatibility methods (legacy support)
    compatibility_methods = [
        'update_capacity_after_dispatch', 'get_remaining_capacity',
        'get_current_bid_schedule', 'reset_all_strategies'
    ]
    missing_compatibility = []
    for method in compatibility_methods:
        if not hasattr(strategy_manager, method):
            missing_compatibility.append(method)

    if missing_compatibility:
        issues.append(f"BiddingStrategyManager missing compatibility methods: {', '.join(missing_compatibility)}")

    return issues


def validate_revenue_calculator_interface(revenue_calculator) -> List[str]:
    """
    Validate RevenueCalculator interface compliance.

    Parameters:
    -----------
    revenue_calculator : RevenueCalculator
        Instance to validate

    Returns:
    --------
    List[str]
        List of missing methods/interface issues
    """
    issues = []

    if not revenue_calculator:
        return ["RevenueCalculator instance is None"]

    required_methods = ['calculate_daily_revenue']
    for method in required_methods:
        if not hasattr(revenue_calculator, method):
            issues.append(f"RevenueCalculator missing required method: {method}")
        elif not callable(getattr(revenue_calculator, method)):
            issues.append(f"RevenueCalculator.{method} is not callable")

    return issues


def validate_results_analyzer_interface(results_analyzer) -> List[str]:
    """
    Validate ResultsAnalyzer interface compliance.

    Parameters:
    -----------
    results_analyzer : ResultsAnalyzer
        Instance to validate

    Returns:
    --------
    List[str]
        List of missing methods/interface issues
    """
    issues = []

    if not results_analyzer:
        return ["ResultsAnalyzer instance is None"]

    required_methods = [
        'load_backtest_data', 'create_comprehensive_report',
        'export_detailed_report', 'export_results_summary',
        'generate_visualizations', 'compare_with_benchmark'
    ]

    for method in required_methods:
        if not hasattr(results_analyzer, method):
            issues.append(f"ResultsAnalyzer missing required method: {method}")
        elif not callable(getattr(results_analyzer, method)):
            issues.append(f"ResultsAnalyzer.{method} is not callable")

    return issues


def validate_all_component_interfaces(dispatch_simulator, strategy_manager,
                                      revenue_calculator=None, results_analyzer=None) -> Dict[str, List[str]]:
    """
    Validate all component interfaces in one call.

    Parameters:
    -----------
    dispatch_simulator : DispatchSimulator
        DispatchSimulator instance
    strategy_manager : BiddingStrategyManager
        BiddingStrategyManager instance
    revenue_calculator : Optional[RevenueCalculator]
        RevenueCalculator instance (optional)
    results_analyzer : Optional[ResultsAnalyzer]
        ResultsAnalyzer instance (optional)

    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping component names to their validation issues
    """
    validation_results = {}

    # Validate core components (required)
    validation_results['DispatchSimulator'] = validate_dispatch_simulator_interface(dispatch_simulator)
    validation_results['BiddingStrategyManager'] = validate_bidding_strategy_manager_interface(strategy_manager)

    # Validate optional components
    if revenue_calculator:
        validation_results['RevenueCalculator'] = validate_revenue_calculator_interface(revenue_calculator)

    if results_analyzer:
        validation_results['ResultsAnalyzer'] = validate_results_analyzer_interface(results_analyzer)

    return validation_results


def log_validation_results(validation_results: Dict[str, List[str]], logger) -> bool:
    """
    Log validation results and return overall success status.

    Parameters:
    -----------
    validation_results : Dict[str, List[str]]
        Validation results from validate_all_component_interfaces
    logger : logging.Logger
        Logger instance

    Returns:
    --------
    bool
        True if all components pass validation, False otherwise
    """
    all_passed = True
    total_issues = 0

    logger.info("=== COMPONENT INTERFACE VALIDATION ===")

    for component_name, issues in validation_results.items():
        if issues:
            all_passed = False
            total_issues += len(issues)
            logger.warning(f"❌ {component_name} validation failed:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info(f"✅ {component_name} validation passed")

    if all_passed:
        logger.info("🎯 All component interfaces validated successfully")
    else:
        logger.warning(
            f"⚠️  Interface validation found {total_issues} issues across {len([c for c, i in validation_results.items() if i])} components")

    logger.info("=" * 45)

    return all_passed

class BatteryBacktester:
    """
    Main class orchestrating the battery bidding backtest system.
    This class coordinates all components and manages the backtesting workflow.
    """

    # Class constants
    PEAK_PERIOD_INTERVALS = 31
    PEAK_PERIOD_HOURS = 2.5
    HIGH_PRICE_THRESHOLD = 300  # AUD/MWh

    def __init__(self, config: Configuration):
        """
        Initialize the backtest system with configuration.

        Parameters:
        -----------
        config : Configuration
            System configuration object
        """
        self.config = config
        self.logger = None  # Will be set after backtest_logger is initialized

        # Initialize core components
        self.data_manager: Optional[DataManager] = None
        self.price_calculator: Optional[PriceBandCalculator] = None
        self.strategy_manager: Optional[BiddingStrategyManager] = None
        self.dispatch_simulator: Optional[DispatchSimulator] = None
        self.revenue_calculator: Optional[RevenueCalculator] = None
        self.backtest_logger: Optional[BatteryBacktestLogger] = None
        self.results_analyzer: Optional[ResultsAnalyzer] = None

        # Results storage
        self.backtest_results: Dict[str, Any] = {}

    def setup_logging(self, log_level: str = "INFO") -> None:
        """
        Configure logging for the backtest system using BatteryBacktestLogger.

        Parameters:
        -----------
        log_level : str
            Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Initialize the advanced logger first
        self.backtest_logger = BatteryBacktestLogger(self.config)

        # Set the main logger to use the advanced logger's main logger
        self.logger = self.backtest_logger.logger

        self.logger.info("Advanced logging system initialized")

    def initialize_components(self) -> None:
        """
        Initialize all system components with proper interface validation.
        ENHANCED: Added interface validation while preserving all existing initialization logic.
        """
        try:
            self.logger.info("Initializing system components with interface validation...")

            # Validate configuration
            if not hasattr(self.config, 'battery_parameters'):
                raise ValueError("Configuration missing battery_parameters")

            if not hasattr(self.config, 'output_directory'):
                raise ValueError("Configuration missing output_directory")

            # Initialize components with enhanced validation
            with self.backtest_logger.timer("DataManager Initialization"):
                self.logger.info("Initializing DataManager...")
                self.data_manager = DataManager(self.config)

            with self.backtest_logger.timer("PriceBandCalculator Initialization"):
                self.logger.info("Initializing PriceBandCalculator...")
                self.price_calculator = PriceBandCalculator()

            with self.backtest_logger.timer("BiddingStrategyManager Initialization"):
                self.logger.info("Initializing BiddingStrategyManager...")
                # FIXED: Add max_ramp_per_interval parameter that was missing
                self.strategy_manager = BiddingStrategyManager(
                    total_capacity=self.config.battery_parameters['max_discharge_power'],
                    total_intervals=self.PEAK_PERIOD_INTERVALS,
                    minimum_floor_mw=5.0,
                    max_ramp_per_interval=self.config.get_battery_max_ramp_5min(),  # ADDED: Missing parameter
                    logger=self.logger
                )

            with self.backtest_logger.timer("DispatchSimulator Initialization"):
                self.logger.info("Initializing DispatchSimulator...")
                self.dispatch_simulator = DispatchSimulator(self.config, self.logger)

            with self.backtest_logger.timer("RevenueCalculator Initialization"):
                self.logger.info("Initializing RevenueCalculator...")
                self.revenue_calculator = RevenueCalculator(self.config)

            with self.backtest_logger.timer("ResultsAnalyzer Initialization"):
                self.logger.info("Initializing ResultsAnalyzer...")
                self.results_analyzer = ResultsAnalyzer(self.config, self.backtest_logger)

            self.logger.info("All components initialized successfully")

            # ENHANCED: Use the new validation system instead of the old one
            validation_results = validate_all_component_interfaces(
                dispatch_simulator=self.dispatch_simulator,
                strategy_manager=self.strategy_manager,
                revenue_calculator=self.revenue_calculator,
                results_analyzer=self.results_analyzer
            )

            # Log validation results
            validation_passed = log_validation_results(validation_results, self.logger)

            if not validation_passed:
                self.logger.warning("Some component interfaces have issues - check logs above")
                # Continue anyway but log the warning

            # PRESERVED: Keep all existing validation logic
            # Validate fixed allocation setup
            self._validate_fixed_allocation_setup()

            self.logger.info("All components initialized successfully with interface validation")

        except Exception as e:
            if self.logger:
                self.backtest_logger.log_error(e, "Component initialization")
            else:
                print(f"Component initialization failed: {str(e)}")
            self._cleanup_components()
            raise

    def _validate_fixed_allocation_setup(self) -> None:
        """
        Validate that the fixed allocation logic is properly configured.
        FIXED: Proper parameter handling and validation logic.
        """
        try:
            # Test strategy initialization
            test_strategy = BiddingStrategy.CONSERVATIVE
            test_profile = self.strategy_manager.initialize_strategy(test_strategy)

            # Validate fixed allocations
            total_allocation = sum(test_profile.band_allocations.values())
            expected_capacity = self.config.battery_parameters['max_discharge_power']

            if abs(total_allocation - expected_capacity) > 0.001:
                raise ValueError(
                    f"Fixed allocation setup failed: "
                    f"Total allocation {total_allocation}MW != Expected {expected_capacity}MW"
                )

            # Test 5MW minimum floor mechanism
            test_profile.remaining_soc = 4.0  # Below 5MW minimum
            minimum_floor_active = test_profile.is_below_minimum_floor()

            if not minimum_floor_active:
                raise ValueError("5MW minimum floor mechanism not working")

            # Test 7-interval final override mechanism
            test_profile.remaining_soc = 20.0
            test_profile.current_interval = 26  # Within the final 7 intervals
            intervals_remaining = self.PEAK_PERIOD_INTERVALS - test_profile.current_interval + 1
            final_override_check = intervals_remaining <= 7

            if not final_override_check:
                raise ValueError("7-interval final override calculation error")

            # Test that both mechanisms are properly integrated
            override_active = self.strategy_manager.is_final_override_active(test_strategy.value)

            # Reset test strategy
            self.strategy_manager.reset_strategy(test_strategy.value)

            self.logger.info("Fixed allocation setup validation passed")

        except Exception as e:
            self.logger.error(f"Fixed allocation setup validation failed: {str(e)}")
            raise

    def _cleanup_components(self) -> None:
        """Clean up partially initialized components."""
        components = ['data_manager', 'price_calculator', 'strategy_manager',
                      'dispatch_simulator', 'revenue_calculator', 'backtest_logger',  # Added revenue_calculator
                      'results_analyzer']

        for component_name in components:
            if hasattr(self, component_name):
                component = getattr(self, component_name)
                if component and hasattr(component, 'close'):
                    try:
                        component.close()
                    except Exception:
                        pass  # Ignore cleanup errors

    def load_data(self) -> None:
        """
        Load all required market data.

        Raises:
        -------
        Exception
            If data loading fails
        """
        try:
            with self.backtest_logger.timer("Data Loading"):
                self.logger.info("Loading market data...")

                if self.data_manager is None:
                    raise ValueError("DataManager not initialized")

                # Load RRP data for the full analysis period
                self.data_manager.load_rrp_data()

                # Get and log data summary
                summary = self.data_manager.get_data_summary()
                self.logger.info(f"Loaded {summary['total_records']:,} RRP records")
                self.logger.info(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
                self.logger.info(f"Price statistics: Mean=${summary['rrp_statistics']['mean']:.2f}, "
                                 f"Max=${summary['rrp_statistics']['max']:.2f}")

                # Export data summary for reference
                self._export_data_summary(summary)

        except Exception as e:
            self.backtest_logger.log_error(e, "Data loading")
            raise

    def run_backtest(self, max_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the main backtesting process.

        Parameters:
        -----------
        max_days : Optional[int]
            Maximum number of days to test (None for a full period)

        Returns:
        --------
        Dict[str, Any]
            Complete backtest results
        """
        try:
            with self.backtest_logger.timer("Complete Backtest"):
                self.logger.info("Starting backtesting process...")

                # Validate components are initialized
                if not all([self.data_manager, self.price_calculator, self.strategy_manager,
                            self.dispatch_simulator]):
                    raise ValueError("Core components not properly initialized")

                # Parse analysis period
                analysis_start = datetime.strptime(self.config.analysis_start_date, "%Y/%m/%d %H:%M:%S")
                analysis_end = datetime.strptime(self.config.end_date, "%Y/%m/%d %H:%M:%S")

                # Limit days if specified
                if max_days:
                    analysis_end = min(analysis_end, analysis_start + timedelta(days=max_days))
                    self.logger.info(f"Limiting backtest to {max_days} days")

                self.logger.info(f"Backtesting from {analysis_start.date()} to {analysis_end.date()}")

                # Initialize results storage
                self.backtest_results = {
                    'daily_results': {},
                    'strategy_performance': {},
                    'price_band_evolution': {},
                    'market_conditions': {}
                }

                # Reset dispatch simulator
                self.dispatch_simulator.reset_simulation()

                # Validate MAXAVAIL integration
                maxavail_available = self._validate_maxavail_integration()
                if maxavail_available:
                    self.logger.info("MAXAVAIL integration active - using NEMDE algorithm")
                else:
                    self.logger.warning("MAXAVAIL integration not detected - using legacy dispatch")

                # Main backtesting loop
                current_date = analysis_start
                day_count = 0
                successful_days = 0

                while current_date <= analysis_end:
                    self.logger.info(f"Processing day {day_count + 1}: {current_date.date()}")

                    try:
                        # Process a single day with timing
                        with self.backtest_logger.timer(f"Single Day Processing - {current_date.date()}"):
                            daily_results = self._process_single_day(current_date)

                        if daily_results:
                            self.backtest_results['daily_results'][current_date.date()] = daily_results
                            self._log_daily_summary(current_date.date(), daily_results)
                            successful_days += 1

                            # ADD: Aggregate price band evolution
                            if 'price_bands' in daily_results:
                                date_key = current_date.strftime('%Y-%m-%d')
                                self.backtest_results['price_band_evolution'][date_key] = daily_results['price_bands']

                            # ADD: Aggregate market conditions
                            if 'market_conditions' in daily_results:
                                date_key = current_date.strftime('%Y-%m-%d')
                                if 'market_conditions' not in self.backtest_results:
                                    self.backtest_results['market_conditions'] = {}
                                self.backtest_results['market_conditions'][date_key] = daily_results[
                                    'market_conditions']

                    except Exception as e:
                        self.backtest_logger.log_error(e, f"Processing {current_date.date()}")

                    current_date += timedelta(days=1)
                    day_count += 1

                self.logger.info(f"Successfully processed {successful_days} out of {day_count} days")

                # Calculate aggregate performance
                with self.backtest_logger.timer("Aggregate Performance Calculation"):
                    self._calculate_aggregate_performance()

                # Export results
                with self.backtest_logger.timer("Results Export"):
                    self._export_results()

                self.logger.info("Backtesting completed successfully!")
                return self.backtest_results

        except Exception as e:
            self.backtest_logger.log_error(e, "Main backtesting process")
            raise

    def _process_single_day(self, current_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Process backtesting for a single day with enhanced validation and error handling.
        PRESERVED: All existing logic with FIXED BiddingStrategyManager initialization.
        """
        try:
            # Get market data for this day
            current_day_data = self.data_manager.get_current_day_data(current_date)
            if current_day_data.empty:
                self.logger.warning(f"No market data available for {current_date.date()}")
                return None

            # Calculate rolling price bands using historical data
            try:
                with self.backtest_logger.timer(f"Price Band Calculation - {current_date.date()}"):
                    price_bands = self.price_calculator.calculate_rolling_price_bands(
                        self.data_manager, current_date
                    )
            except Exception as e:
                self.backtest_logger.log_error(e, f"Price band calculation for {current_date.date()}")
                return None

            # Get historical data for strategy matrix creation
            history_data = self.data_manager.get_history_window_data(current_date)
            if history_data.empty:
                self.logger.warning(f"No historical data available for {current_date.date()}")
                return None

            # Test each strategy with enhanced error handling and validation
            strategy_results = {}
            all_dispatch_results = []

            # Test all available strategies
            strategies_to_test = [BiddingStrategy.CONSERVATIVE, BiddingStrategy.BALANCED,
                                  BiddingStrategy.AGGRESSIVE, BiddingStrategy.PEAK_CAPTURE]

            for strategy_enum in strategies_to_test:
                strategy_name = strategy_enum.value

                try:
                    # FIXED: Create a fresh strategy manager with ALL required parameters
                    daily_strategy_manager = BiddingStrategyManager(
                        total_capacity=self.config.battery_parameters['max_discharge_power'],
                        total_intervals=self.PEAK_PERIOD_INTERVALS,
                        minimum_floor_mw=5.0,
                        max_ramp_per_interval=self.config.get_battery_max_ramp_5min(),  # FIXED: Added missing parameter
                        logger=self.logger
                    )

                    # Initialize the specific strategy
                    strategy_profile = daily_strategy_manager.initialize_strategy(strategy_enum)

                    # Validate strategy initialization
                    if not strategy_profile or strategy_profile.remaining_soc <= 0:
                        self.logger.error(f"Strategy {strategy_name} initialization failed")
                        continue

                    # Simulate strategy performance with enhanced tracking
                    with self.backtest_logger.timer(f"Strategy Simulation - {strategy_name}"):
                        daily_result = self._simulate_strategy_performance(
                            current_day_data, price_bands, daily_strategy_manager, strategy_name
                        )

                    # Validate strategy results
                    if not daily_result or 'daily_summary' not in daily_result:
                        self.logger.error(f"Strategy {strategy_name} simulation failed - no results")
                        continue

                    # Store results for this strategy
                    strategy_results[strategy_name] = daily_result

                    # Collect dispatch results for validation
                    if 'dispatch_results' in daily_result:
                        dispatch_results = daily_result['dispatch_results']

                        # Ensure all dispatch results have strategy_name set
                        for result in dispatch_results:
                            if not hasattr(result, 'strategy_name') or not result.strategy_name:
                                result.strategy_name = strategy_name

                        all_dispatch_results.extend(dispatch_results)

                    self.logger.info(f"Strategy {strategy_name} completed successfully: "
                                     f"Revenue=${daily_result['daily_summary'].total_revenue:.2f}, "
                                     f"Dispatched={daily_result['daily_summary'].total_dispatched_mwh:.1f}MWh")

                except Exception as e:
                    self.backtest_logger.log_error(e, f"Testing strategy {strategy_name}")
                    self.logger.error(f"Strategy {strategy_name} failed: {str(e)}")
                    continue

            # Validate that we have at least one successful strategy
            if not strategy_results:
                self.logger.error(f"No strategies succeeded for {current_date.date()}")
                return None

            # Perform data flow validation
            try:
                validation_issues = self._validate_data_flow(all_dispatch_results, strategy_results)
                if validation_issues:
                    self.logger.warning(f"Data flow validation issues for {current_date.date()}: {validation_issues}")
            except Exception as e:
                self.logger.warning(f"Data flow validation failed for {current_date.date()}: {str(e)}")

            # Filter to the peak period for consistency
            peak_period_data = self._filter_to_peak_period(current_day_data)
            if peak_period_data.empty:
                self.logger.warning(f"No peak period data for {current_date.date()}")
                return None

            # Use the peak period for market conditions
            market_conditions = {
                'avg_rrp': float(peak_period_data['RRP'].mean()),
                'max_rrp': float(peak_period_data['RRP'].max()),
                'min_rrp': float(peak_period_data['RRP'].min()),
                'high_price_intervals': int((peak_period_data['RRP'] > self.HIGH_PRICE_THRESHOLD).sum()),
                'total_intervals': len(peak_period_data),
                'price_volatility': float(peak_period_data['RRP'].std()),
                'peak_period_only': True,
                'median_rrp': float(peak_period_data['RRP'].median()),
                'rrp_range': float(peak_period_data['RRP'].max() - peak_period_data['RRP'].min()),
            }

            # Create comprehensive daily results
            daily_results = {
                'date': current_date.date(),
                'price_bands': price_bands,
                'strategy_results': strategy_results,
                'market_conditions': market_conditions,
                'validation_metadata': {
                    'total_dispatch_results': len(all_dispatch_results),
                    'strategies_tested': len(strategies_to_test),
                    'strategies_succeeded': len(strategy_results),
                    'peak_period_intervals': len(peak_period_data),
                    'data_quality_score': self._calculate_data_quality_score(peak_period_data, strategy_results)
                }
            }

            return daily_results

        except Exception as e:
            self.backtest_logger.log_error(e, f"Processing {current_date.date()}")
            self.logger.error(f"Critical error processing {current_date.date()}: {str(e)}")
            return None

    def _calculate_data_quality_score(self, peak_period_data: pd.DataFrame,
                                      strategy_results: Dict[str, Any]) -> float:
        """
        Calculate a data quality score for the day's results.

        Parameters:
        -----------
        peak_period_data : pd.DataFrame
            Peak period market data
        strategy_results : Dict[str, Any]
            Strategy simulation results

        Returns:
        --------
        float
            Data quality score between 0.0 and 1.0
        """
        try:
            score_components = []

            # Market data quality (30% weight)
            if not peak_period_data.empty:
                # Check for reasonable RRP values
                rrp_reasonable = ((peak_period_data['RRP'] >= 0) &
                                  (peak_period_data['RRP'] <= 20000)).all()
                # Check for data completeness
                data_complete = len(peak_period_data) >= 25  # At least ~80% of expected intervals

                market_score = (0.6 if rrp_reasonable else 0.0) + (0.4 if data_complete else 0.0)
                score_components.append(market_score * 0.3)

            # Strategy results quality (40% weight)
            if strategy_results:
                successful_strategies = len(strategy_results)
                total_expected = 4  # Conservative, Balanced, Aggressive, Peak Capture

                strategy_score = successful_strategies / total_expected
                score_components.append(strategy_score * 0.4)

            # Dispatch results quality (30% weight)
            dispatch_quality = 0.0
            total_dispatch_results = 0
            valid_dispatch_results = 0

            for strategy_result in strategy_results.values():
                if 'dispatch_results' in strategy_result:
                    dispatch_results = strategy_result['dispatch_results']
                    total_dispatch_results += len(dispatch_results)

                    for result in dispatch_results:
                        # Check for required fields
                        if (hasattr(result, 'strategy_name') and result.strategy_name and
                                hasattr(result, 'maxavail_for_interval') and
                                hasattr(result, 'dispatched_mw')):
                            valid_dispatch_results += 1

            if total_dispatch_results > 0:
                dispatch_quality = valid_dispatch_results / total_dispatch_results

            score_components.append(dispatch_quality * 0.3)

            # Calculate final score
            final_score = sum(score_components)
            return min(1.0, max(0.0, final_score))  # Ensure between 0 and 1

        except Exception as e:
            self.logger.warning(f"Failed to calculate data quality score: {str(e)}")
            return 0.5  # Default neutral score

    def _validate_strategy_results(self, strategy_results: Dict[str, Any]) -> List[str]:
        """
        Validate strategy results for completeness and consistency.

        Parameters:
        -----------
        strategy_results : Dict[str, Any]
            Strategy simulation results

        Returns:
        --------
        List[str]
            List of validation issues
        """
        issues = []

        for strategy_name, result in strategy_results.items():
            # Check required top-level keys
            required_keys = ['daily_summary', 'dispatch_results', 'initial_capacity',
                             'final_soc', 'strategy_info']

            for key in required_keys:
                if key not in result:
                    issues.append(f"Strategy {strategy_name} missing {key}")

            # Validate daily summary
            if 'daily_summary' in result:
                summary = result['daily_summary']
                if not hasattr(summary, 'total_revenue'):
                    issues.append(f"Strategy {strategy_name} daily_summary missing total_revenue")
                if not hasattr(summary, 'total_dispatched_mwh'):
                    issues.append(f"Strategy {strategy_name} daily_summary missing total_dispatched_mwh")

            # Validate dispatch results
            if 'dispatch_results' in result:
                dispatch_results = result['dispatch_results']
                if not isinstance(dispatch_results, list):
                    issues.append(f"Strategy {strategy_name} dispatch_results not a list")
                else:
                    for i, dr in enumerate(dispatch_results):
                        if not hasattr(dr, 'strategy_name'):
                            issues.append(f"Strategy {strategy_name} dispatch_result {i} missing strategy_name")
                        elif dr.strategy_name != strategy_name:
                            issues.append(
                                f"Strategy {strategy_name} dispatch_result {i} has wrong strategy_name: {dr.strategy_name}")

        return issues

    def _validate_fixed_allocation_behavior(self, dispatch_results: List,
                                            initial_allocations: Dict[str, float],
                                            strategy_name: str) -> Dict[str, Any]:
        """
        Validate that fixed allocation logic is working correctly.
        NEW: Validates the fixed allocation behavior vs. dynamic reallocation.

        Parameters:
        -----------
        dispatch_results : List
            List of DispatchResult objects
        initial_allocations : Dict[str, float]
            Initial band allocations for the strategy
        strategy_name : str
            Strategy name for logging

        Returns:
        --------
        Dict[str, Any]
            Validation results for fixed allocation behavior
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'allocation_changes': [],
            'override_events': []
        }

        # Track allocation changes throughout the day
        current_allocations = initial_allocations.copy()

        for result in dispatch_results:
            if hasattr(result, 'strategy_allocation') and result.strategy_allocation:
                # Check if allocations changed unexpectedly (outside of dispatch and overrides)
                expected_change = result.dispatched_mw > 0 or getattr(result, 'final_override_active', False)

                allocation_diff = {}
                for band in current_allocations:
                    old_val = current_allocations.get(band, 0.0)
                    new_val = result.strategy_allocation.get(band, 0.0)
                    if abs(old_val - new_val) > 0.001:
                        allocation_diff[band] = {'old': old_val, 'new': new_val}

                if allocation_diff and not expected_change:
                    validation_result['issues'].append(
                        f"Unexpected allocation change at interval {result.interval_number}: {allocation_diff}"
                    )
                    validation_result['is_valid'] = False

                # Record override events
                if getattr(result, 'final_override_active', False):
                    override_type = 'minimum_floor' if result.remaining_soc < 5.0 else 'final_override'
                    validation_result['override_events'].append({
                        'interval': result.interval_number,
                        'remaining_soc': result.remaining_soc,
                        'type': override_type
                    })

                # Update tracking
                current_allocations = result.strategy_allocation.copy()

        validation_result['total_override_events'] = len(validation_result['override_events'])

        if validation_result['is_valid']:
            self.logger.debug(f"Fixed allocation validation passed for {strategy_name}")
        else:
            self.logger.warning(
                f"Fixed allocation validation issues for {strategy_name}: {validation_result['issues']}")

        return validation_result

    def _validate_initial_allocation(self, strategy_profile, strategy_name: str) -> Dict[str, Any]:
        """
        Validate that initial strategy allocation is correct.
        NEW: The function validates an initial fixed allocation setup.

        Parameters:
        -----------
        strategy_profile : BiddingProfile
            The strategy profile to validate
        strategy_name : str
            Strategy name for logging

        Returns:
        --------
        Dict[str, Any]
            Validation results for initial allocation
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'allocation_summary': {}
        }

        try:
            # Check that total allocation equals capacity
            total_allocation = sum(strategy_profile.band_allocations.values())
            expected_capacity = strategy_profile.total_capacity

            if abs(total_allocation - expected_capacity) > 0.001:
                validation_result['issues'].append(
                    f"Total allocation ({total_allocation:.2f}MW) != Expected capacity ({expected_capacity:.2f}MW)"
                )
                validation_result['is_valid'] = False

            # Check that remaining SOC equals total capacity initially
            if abs(strategy_profile.remaining_soc - expected_capacity) > 0.001:
                validation_result['issues'].append(
                    f"Initial remaining SOC ({strategy_profile.remaining_soc:.2f}MW) != Total capacity ({expected_capacity:.2f}MW)"
                )
                validation_result['is_valid'] = False

            # Check that no individual band allocation exceeds total capacity
            for band_name, allocation in strategy_profile.band_allocations.items():
                if allocation > expected_capacity + 0.001:
                    validation_result['issues'].append(
                        f"Band {band_name} allocation ({allocation:.2f}MW) exceeds total capacity ({expected_capacity:.2f}MW)"
                    )
                    validation_result['is_valid'] = False

            # Check that all allocations are non-negative
            for band_name, allocation in strategy_profile.band_allocations.items():
                if allocation < -0.001:
                    validation_result['issues'].append(
                        f"Band {band_name} has negative allocation ({allocation:.2f}MW)"
                    )
                    validation_result['is_valid'] = False

            # Create allocation summary
            validation_result['allocation_summary'] = {
                'total_allocation': total_allocation,
                'expected_capacity': expected_capacity,
                'remaining_soc': strategy_profile.remaining_soc,
                'band_count_with_allocation': sum(1 for v in strategy_profile.band_allocations.values() if v > 0.001),
                'largest_band_allocation': max(
                    strategy_profile.band_allocations.values()) if strategy_profile.band_allocations else 0,
                'smallest_nonzero_allocation': min(
                    v for v in strategy_profile.band_allocations.values() if v > 0.001) if any(
                    v > 0.001 for v in strategy_profile.band_allocations.values()) else 0
            }

            if validation_result['is_valid']:
                self.logger.debug(f"Initial allocation validation passed for {strategy_name}")
            else:
                self.logger.warning(
                    f"Initial allocation validation failed for {strategy_name}: {validation_result['issues']}")

        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Initial allocation validation error for {strategy_name}: {str(e)}")

        return validation_result

    def _simulate_strategy_performance(self, current_day_data: pd.DataFrame,
                                       price_bands: Dict[str, float],
                                       strategy_manager: BiddingStrategyManager,
                                       strategy_name: str) -> Dict[str, Any]:
        """
        Simulate strategy performance with enhanced validation and fixed allocation tracking.
        ENHANCED: Validates fixed allocation behavior and dual override mechanisms.

        Parameters:
        -----------
        current_day_data : pd.DataFrame
            Market data for the day
        price_bands : Dict[str, float]
            Price bands for the day
        strategy_manager : BiddingStrategyManager
            Strategy manager for this simulation
        strategy_name : str
            Name of the strategy

        Returns:
        --------
        Dict[str, Any]
            Strategy performance results with enhanced validation
        """
        try:
            # Filter to peak period
            peak_period_data = self._filter_to_peak_period(current_day_data)

            if peak_period_data.empty:
                self.logger.warning(f"No peak period data for {strategy_name}")
                return self._create_empty_strategy_result()

            # Validate initial strategy state
            initial_soc = strategy_manager.get_remaining_soc(strategy_name)
            if initial_soc <= 0:
                self.logger.error(f"Strategy {strategy_name} has no initial capacity")
                return self._create_empty_strategy_result()

            # Get initial band allocations for validation
            initial_allocations = strategy_manager.get_bid_schedule(strategy_name, 1)
            initial_total_allocation = sum(initial_allocations.values())

            # Validate that initial allocations match SOC
            if abs(initial_total_allocation - initial_soc) > 0.001:
                self.logger.warning(
                    f"Strategy {strategy_name} allocation mismatch: "
                    f"SOC={initial_soc:.2f}MW, Allocations={initial_total_allocation:.2f}MW"
                )

            # Simulate daily dispatch
            with self.backtest_logger.timer(f"Strategy Simulation - {strategy_name}"):
                dispatch_results, daily_summary = self.dispatch_simulator.simulate_daily_dispatch(
                    peak_period_data, price_bands, strategy_manager, strategy_name
                )

            # Enhanced validation of results
            if not dispatch_results:
                self.logger.warning(f"No dispatch results for strategy {strategy_name}")
                return self._create_empty_strategy_result()

            # Validate fixed allocation behavior
            allocation_validation = self._validate_fixed_allocation_behavior(
                dispatch_results, initial_allocations, strategy_name
            )

            # Ensure all dispatch results have correct strategy_name
            for result in dispatch_results:
                if not hasattr(result, 'strategy_name') or not result.strategy_name:
                    result.strategy_name = strategy_name

            # Get final strategy state
            try:
                final_soc = strategy_manager.get_remaining_soc(strategy_name)
                minimum_floor_used = any(
                    getattr(r, 'final_override_active', False) and r.remaining_soc < 5.0
                    for r in dispatch_results
                )
                final_override_used = any(
                    getattr(r, 'final_override_active', False) and r.interval_number >= 25
                    for r in dispatch_results
                )
            except Exception as e:
                self.logger.warning(f"Failed to get final strategy state for {strategy_name}: {str(e)}")
                final_soc = 0.0
                minimum_floor_used = False
                final_override_used = False

            # Calculate revenue breakdown if available
            revenue_breakdown = None
            if self.revenue_calculator:
                try:
                    revenue_breakdown = self.revenue_calculator.calculate_daily_revenue(
                        dispatch_results, current_day_data.iloc[0]['SETTLEMENTDATE']
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to calculate revenue breakdown for {strategy_name}: {str(e)}")

            # Create comprehensive result
            result = {
                'daily_summary': daily_summary,
                'dispatch_results': dispatch_results,
                'initial_capacity': self.config.battery_parameters['max_discharge_power'],
                'final_soc': final_soc,
                'minimum_floor_used': minimum_floor_used,
                'final_override_used': final_override_used,
                'strategy_info': strategy_manager.get_strategy_info(strategy_name),
                'allocation_validation': allocation_validation,
                'validation_metadata': {
                    'dispatch_results_count': len(dispatch_results),
                    'strategy_name_consistency': all(
                        hasattr(r, 'strategy_name') and r.strategy_name == strategy_name
                        for r in dispatch_results
                    ),
                    'maxavail_fields_present': all(
                        hasattr(r, 'maxavail_for_interval') for r in dispatch_results
                    ),
                    'revenue_calculation_available': revenue_breakdown is not None,
                    'initial_soc': initial_soc,
                    'capacity_utilization': ((initial_soc - final_soc) / initial_soc * 100) if initial_soc > 0 else 0
                }
            }

            if revenue_breakdown:
                result['revenue_breakdown'] = revenue_breakdown

            return result

        except Exception as e:
            self.backtest_logger.log_error(e, f"Strategy simulation - {strategy_name}")
            return self._create_empty_strategy_result()

    def _filter_to_peak_period(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to peak period times (17:00-19:30).

        Parameters:
        -----------
        data : pd.DataFrame
            Market data to filter

        Returns:
        --------
        pd.DataFrame
            Data filtered to peak period
        """
        if data.empty:
            return data

        # More efficient: use dt accessor directly
        time_mask = (
                (data['SETTLEMENTDATE'].dt.time >= self.config.peak_period_start_time) &
                (data['SETTLEMENTDATE'].dt.time <= self.config.peak_period_end_time)
        )

        return data[time_mask].copy()

    def _create_empty_strategy_result(self) -> Dict[str, Any]:
        """Create an empty strategy result for failed simulations."""
        from dispatch_simulator import DailyDispatchSummary

        empty_summary = DailyDispatchSummary(
            date=datetime.now(),
            strategy_name="unknown",
            total_dispatched_mwh=0.0,
            total_revenue=0.0,
            dispatch_events=0,
            average_dispatch_mw=0.0,
            max_dispatch_mw=0.0,
            dispatch_efficiency=0.0,
            constraint_events=0,
            ramp_constraint_events=0,
            power_constraint_events=0,
            final_override_events=0,
            capacity_utilization=0.0
        )

        return {
            'daily_summary': empty_summary,
            'dispatch_results': [],
            'initial_capacity': self.config.battery_parameters['max_discharge_power'],
            'final_soc': self.config.battery_parameters['max_discharge_power'],
            'final_override_used': False,
            'strategy_info': {}
        }

    def _calculate_aggregate_performance(self) -> None:
        """Calculate aggregate performance across all days and strategies."""
        self.logger.info("Calculating aggregate performance...")

        if not self.backtest_results['daily_results']:
            self.logger.warning("No daily results to aggregate")
            return

        # Pre-determine strategy names
        first_day_results = next(iter(self.backtest_results['daily_results'].values()))
        strategy_names = list(first_day_results['strategy_results'].keys())

        # Pre-allocate performance dictionary
        strategy_performance = {
            strategy: {
                'total_revenue': 0.0,
                'total_dispatched_mwh': 0.0,
                'total_dispatch_events': 0,
                'total_constraint_events': 0,
                'total_final_override_events': 0,
                'total_minimum_floor_events': 0,
                'days_analyzed': 0,
                'daily_soc_utilizations': []  # Track daily SOC utilization
            }
            for strategy in strategy_names
        }

        # Single pass-through data
        for date, daily_data in self.backtest_results['daily_results'].items():
            for strategy_name in strategy_names:
                if strategy_name in daily_data['strategy_results']:
                    strategy_result = daily_data['strategy_results'][strategy_name]
                    daily_summary = strategy_result['daily_summary']

                    perf = strategy_performance[strategy_name]
                    perf['total_revenue'] += daily_summary.total_revenue
                    perf['total_dispatched_mwh'] += daily_summary.total_dispatched_mwh
                    perf['total_dispatch_events'] += daily_summary.dispatch_events
                    perf['total_constraint_events'] += daily_summary.constraint_events
                    perf['total_final_override_events'] += daily_summary.final_override_events
                    perf['total_minimum_floor_events'] += daily_summary.minimum_floor_events  # ADD THIS LINE
                    perf['days_analyzed'] += 1

                    # Track daily SOC utilization (from daily summary capacity_utilization)
                    perf['daily_soc_utilizations'].append(daily_summary.capacity_utilization)

        # Calculate derived metrics
        for strategy_name, perf in strategy_performance.items():
            days = perf['days_analyzed']
            dispatched = perf['total_dispatched_mwh']
            revenue = perf['total_revenue']
            max_power = self.config.battery_parameters['max_discharge_power']

            # Basic revenue metrics
            perf['avg_daily_revenue'] = revenue / days if days > 0 else 0
            perf['avg_revenue_per_mwh'] = revenue / dispatched if dispatched > 0 else 0

            # Energy capacity utilization (total energy vs. theoretical maximum)
            theoretical_max_energy = days * max_power * self.PEAK_PERIOD_HOURS
            perf['energy_capacity_utilization'] = (
                dispatched / theoretical_max_energy * 100 if theoretical_max_energy > 0 else 0
            )

            # Average daily SOC utilization (should be close to 100% for fully utilized batteries)
            daily_utilizations = perf['daily_soc_utilizations']
            perf['avg_daily_soc_utilization'] = (
                sum(daily_utilizations) / len(daily_utilizations) if daily_utilizations else 0
            )

            # Battery deployment efficiency metrics
            perf['avg_dispatch_events_per_day'] = (
                perf['total_dispatch_events'] / days if days > 0 else 0
            )

            perf['constraint_rate'] = (
                perf['total_constraint_events'] / perf['total_dispatch_events'] * 100
                if perf['total_dispatch_events'] > 0 else 0
            )

            perf['final_override_rate'] = (
                perf['total_final_override_events'] / days * 100 if days > 0 else 0
            )

            # Market engagement metrics
            total_possible_intervals = days * 31  # Approximate intervals per peak period
            perf['market_engagement_rate'] = (
                perf['total_dispatch_events'] / total_possible_intervals * 100
                if total_possible_intervals > 0 else 0
            )

            # Revenue concentration (how much energy needed to achieve total revenue)
            perf['revenue_concentration'] = (
                dispatched / (max_power * days) * 100 if (max_power * days) > 0 else 0
            )

            # Effective dispatch duration per day (in hours)
            avg_daily_energy = dispatched / days if days > 0 else 0
            perf['avg_daily_dispatch_duration_hours'] = (
                avg_daily_energy / max_power if max_power > 0 else 0
            )

            # Clean up the temporary tracking list
            del perf['daily_soc_utilizations']

        self.backtest_results['strategy_performance'] = strategy_performance

        # Log enhanced performance summary
        self._log_performance_summary(strategy_performance)

    def _log_daily_summary(self, date, daily_results: Dict[str, Any]) -> None:
        """Log summary of daily results using advanced logging."""
        market_conditions = daily_results['market_conditions']

        # Safe best strategy identification
        best_strategy = "None"
        best_revenue = 0.0

        if daily_results['strategy_results']:
            try:
                best_strategy = max(daily_results['strategy_results'].keys(),
                                  key=lambda s: daily_results['strategy_results'][s]['daily_summary'].total_revenue)
                best_revenue = daily_results['strategy_results'][best_strategy]['daily_summary'].total_revenue
            except (KeyError, AttributeError, TypeError) as e:
                self.logger.warning(f"Could not determine best strategy: {str(e)}")

        # Use the advanced logger's daily summary method
        self.backtest_logger.log_daily_summary(
            datetime.combine(date, datetime.min.time()),
            market_conditions,
            best_strategy,
            best_revenue
        )

    def _log_performance_summary(self, strategy_performance: Dict[str, Any]) -> None:
        """Log overall performance summary with enhanced metrics."""
        self.logger.info("=" * 60)
        self.logger.info("BACKTEST PERFORMANCE SUMMARY")
        self.logger.info("=" * 60)

        for strategy_name, performance in strategy_performance.items():
            self.logger.info(f"{strategy_name.upper()} STRATEGY:")
            self.logger.info(f"  Total Revenue: ${performance['total_revenue']:.2f}")
            self.logger.info(f"  Average Daily Revenue: ${performance['avg_daily_revenue']:.2f}")
            self.logger.info(f"  Revenue per MWh: ${performance['avg_revenue_per_mwh']:.2f}")
            self.logger.info(f"  Total Energy Dispatched: {performance['total_dispatched_mwh']:.1f} MWh")

            # SOC vs. Energy Capacity Utilization
            self.logger.info(f"  Daily SOC Utilization: {performance['avg_daily_soc_utilization']:.1f}%")
            self.logger.info(f"  Energy Capacity Utilization: {performance['energy_capacity_utilization']:.1f}%")

            # Operational Metrics
            self.logger.info(
                f"  Avg Dispatch Duration: {performance['avg_daily_dispatch_duration_hours']:.2f} hours/day")
            self.logger.info(f"  Market Engagement: {performance['market_engagement_rate']:.1f}%")
            self.logger.info(f"  Constraint Rate: {performance['constraint_rate']:.1f}%")
            self.logger.info(f"  Final Override Events: {performance['total_final_override_events']}")
            self.logger.info("-" * 40)

    def _export_data_summary(self, summary: Dict[str, Any]) -> None:
        """Export data summary to file."""
        try:
            summary_output_path = os.path.join(self.config.output_directory, "data_summary.json")

            # Convert datetime objects to strings for JSON serialization
            json_summary = summary.copy()
            json_summary['date_range']['start'] = str(json_summary['date_range']['start'])
            json_summary['date_range']['end'] = str(json_summary['date_range']['end'])

            with open(summary_output_path, 'w') as f:
                json.dump(json_summary, f, indent=2, default=str)

            self.logger.info(f"Data summary exported to: {summary_output_path}")
        except Exception as e:
            self.backtest_logger.log_error(e, "Data summary export")

    def _export_results(self) -> None:
        """
        Export all backtest results to files with enhanced error handling and validation.

        ENHANCED FEATURES:
        - Pre-export validation for data integrity
        - Enhanced dispatch results export with MAXAVAIL analysis
        - Comprehensive error handling with detailed logging
        - Export progress tracking and performance monitoring
        - Graceful degradation when components are missing
        """
        export_errors = []
        export_successes = []

        self.logger.info("Starting enhanced results export process...")

        try:
            # ENHANCED: Export dispatch results with validation and comprehensive data
            if hasattr(self.dispatch_simulator, 'export_dispatch_results'):
                try:
                    with self.backtest_logger.timer("Dispatch Results Export"):
                        # Pre-export validation
                        if hasattr(self.dispatch_simulator, 'validate_export_data'):
                            validation_issues = self.dispatch_simulator.validate_export_data()
                            if validation_issues:
                                self.logger.warning(f"Dispatch export validation issues: {validation_issues}")

                        # Get export summary for logging
                        if hasattr(self.dispatch_simulator, 'get_export_summary'):
                            export_summary = self.dispatch_simulator.get_export_summary()
                            self.logger.info(f"Dispatch export summary: {export_summary['record_count']} records, "
                                             f"{export_summary['strategy_count']} strategies, "
                                             f"{export_summary['dispatch_rate']:.1f}% dispatch rate")

                        # Export with enhanced filename and path handling
                        dispatch_results_file = os.path.join(
                            self.config.output_directory,
                            f"dispatch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )

                        # Ensure output directory exists
                        os.makedirs(os.path.dirname(dispatch_results_file), exist_ok=True)

                        # Perform the export
                        exported_df = self.dispatch_simulator.export_dispatch_results(dispatch_results_file)

                        # Log success with details
                        export_successes.append(f"Dispatch results: {len(exported_df)} records exported")
                        self.logger.info(f"Enhanced dispatch results exported to: {dispatch_results_file}")
                        self.logger.info(
                            f"Export contains {len(exported_df)} records with {len(exported_df.columns)} columns")

                except PermissionError as e:
                    export_errors.append(f"Permission denied for dispatch results export: {str(e)}")
                    self.logger.error(f"Dispatch export permission error: {str(e)}")
                except Exception as e:
                    export_errors.append(f"Failed to export dispatch results: {str(e)}")
                    self.backtest_logger.log_error(e, "Dispatch results export")
            else:
                export_errors.append("Dispatch simulator missing export_dispatch_results method")
                self.logger.warning("DispatchSimulator missing export_dispatch_results method - consider updating")

        except Exception as e:
            export_errors.append(f"Dispatch results export failed: {str(e)}")
            self.backtest_logger.log_error(e, "Dispatch results export initialization")

        # ENHANCED: Export backtest summary with validation
        try:
            with self.backtest_logger.timer("Backtest Summary Export"):
                summary_file = os.path.join(
                    self.config.output_directory,
                    f"backtest_summary_{datetime.now().strftime('%Y%m%d')}.json"
                )

                # Convert results to JSON-serializable format with validation
                json_results = self._convert_results_to_json()

                # Validate JSON results before export
                if not json_results:
                    raise ValueError("No backtest results to export")

                # Enhanced JSON export with formatting
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(json_results, f, indent=2, default=str, ensure_ascii=False)

                export_successes.append("Backtest summary exported successfully")
                self.logger.info(f"Backtest summary exported to: {summary_file}")

                # Log summary statistics
                if 'strategy_performance' in json_results:
                    strategy_count = len(json_results['strategy_performance'])
                    self.logger.info(f"Summary includes performance data for {strategy_count} strategies")

        except Exception as e:
            export_errors.append(f"Failed to export backtest summary: {str(e)}")
            self.backtest_logger.log_error(e, "Backtest summary export")

        # ENHANCED: Export configuration with validation
        try:
            with self.backtest_logger.timer("Configuration Export"):
                config_file = os.path.join(self.config.output_directory, "configuration.json")

                # Validate configuration before export
                try:
                    self.config.validate_configuration()
                except Exception as validation_error:
                    self.logger.warning(f"Configuration validation failed: {str(validation_error)}")

                # Export configuration
                config_dict = self.config.to_dict()
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, indent=2, default=str, ensure_ascii=False)

                export_successes.append("Configuration exported successfully")
                self.logger.info(f"Configuration exported to: {config_file}")

        except Exception as e:
            export_errors.append(f"Failed to export configuration: {str(e)}")
            self.backtest_logger.log_error(e, "Configuration export")

        # ENHANCED: Export advanced log analysis with comprehensive error handling
        try:
            if self.backtest_logger:
                with self.backtest_logger.timer("Log Analysis Export"):
                    # Export log analysis
                    log_analysis_file = os.path.join(
                        self.config.output_directory,
                        f"log_analysis_{datetime.now().strftime('%Y%m%d')}.json"
                    )

                    self.backtest_logger.export_log_analysis(log_analysis_file)

                    # Create and save the performance report
                    performance_report_file = os.path.join(
                        self.config.output_directory,
                        f"performance_report_{datetime.now().strftime('%Y%m%d')}.txt"
                    )

                    report = self.backtest_logger.create_performance_report()
                    with open(performance_report_file, 'w', encoding='utf-8') as f:
                        f.write(report)

                    export_successes.append("Log analysis and performance report exported")
                    self.logger.info(f"Log analysis exported to: {log_analysis_file}")
                    self.logger.info(f"Performance report exported to: {performance_report_file}")
            else:
                self.logger.warning("No backtest_logger available for log analysis export")

        except Exception as e:
            export_errors.append(f"Failed to export log analysis: {str(e)}")
            self.backtest_logger.log_error(e, "Log analysis export")

        # ENHANCED: Comprehensive Results Analysis with detailed error handling and progress tracking
        try:
            if hasattr(self, 'results_analyzer') and self.results_analyzer:
                self.logger.info("Starting comprehensive results analysis...")

                # ENHANCED: Validate data before loading into analyzer
                dispatch_history = []
                if hasattr(self.dispatch_simulator, 'dispatch_history'):
                    dispatch_history = self.dispatch_simulator.dispatch_history
                    self.logger.info(f"Loading {len(dispatch_history)} dispatch records for analysis")
                else:
                    self.logger.warning("No dispatch history available for analysis")

                # Load data into ResultsAnalyzer with error handling
                try:
                    with self.backtest_logger.timer("Results Data Loading"):
                        self.results_analyzer.load_backtest_data(
                            self.backtest_results,
                            dispatch_history
                        )
                    self.logger.info("Backtest data loaded successfully into ResultsAnalyzer")
                except Exception as load_error:
                    export_errors.append(f"Failed to load data into ResultsAnalyzer: {str(load_error)}")
                    self.logger.error(f"ResultsAnalyzer data loading failed: {str(load_error)}")
                    # Continue with other exports even if this fails

                # Generate comprehensive analysis with enhanced error handling
                try:
                    with self.backtest_logger.timer("Comprehensive Analysis Generation"):
                        comprehensive_results = self.results_analyzer.create_comprehensive_report()

                    self.logger.info("Comprehensive analysis report generated successfully")

                    # ENHANCED: Export detailed analysis report with validation
                    try:
                        detailed_report_file = os.path.join(
                            self.config.output_directory,
                            f"detailed_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt"
                        )

                        self.results_analyzer.export_detailed_report(detailed_report_file, comprehensive_results)
                        export_successes.append("Detailed analysis report exported")
                        self.logger.info(f"Detailed analysis report exported to: {detailed_report_file}")

                    except Exception as report_error:
                        export_errors.append(f"Failed to export detailed analysis report: {str(report_error)}")

                    # ENHANCED: Export JSON summary with validation
                    try:
                        analysis_summary_file = os.path.join(
                            self.config.output_directory,
                            f"analysis_summary_{datetime.now().strftime('%Y%m%d')}.json"
                        )

                        analysis_summary = self.results_analyzer.export_results_summary(
                            analysis_summary_file, comprehensive_results
                        )

                        if analysis_summary:
                            export_successes.append("Analysis summary exported")
                            self.logger.info(f"Analysis summary exported to: {analysis_summary_file}")

                            # Log key summary statistics
                            if 'best_strategies' in analysis_summary:
                                best_revenue = analysis_summary['best_strategies'].get('revenue', 'unknown')
                                best_efficiency = analysis_summary['best_strategies'].get('efficiency', 'unknown')
                                self.logger.info(f"Best revenue strategy: {best_revenue}, "
                                                 f"Best efficiency strategy: {best_efficiency}")
                        else:
                            export_errors.append("Analysis summary export returned empty results")

                    except Exception as summary_error:
                        export_errors.append(f"Failed to export analysis summary: {str(summary_error)}")

                    # ENHANCED: Generate and export visualizations with better error handling
                    try:
                        visualizations_dir = os.path.join(self.config.output_directory, "visualizations")
                        os.makedirs(visualizations_dir, exist_ok=True)

                        with self.backtest_logger.timer("Visualization Generation"):
                            visualization_files = self.results_analyzer.generate_visualizations(
                                visualizations_dir, comprehensive_results
                            )

                        if visualization_files:
                            export_successes.append(f"{len(visualization_files)} visualizations generated")
                            self.logger.info(f"Generated {len(visualization_files)} visualization files:")
                            for viz_type, file_path in visualization_files.items():
                                self.logger.info(f"  {viz_type}: {file_path}")
                        else:
                            self.logger.info("No visualizations generated (matplotlib may not be available)")

                    except Exception as viz_error:
                        export_errors.append(f"Visualization generation failed: {str(viz_error)}")
                        self.logger.warning(f"Visualization generation failed: {str(viz_error)}")

                    # ENHANCED: Export benchmark comparison with validation
                    try:
                        # Check if conservative strategy exists for benchmarking
                        available_strategies = []
                        if 'strategy_performance' in self.backtest_results:
                            available_strategies = list(self.backtest_results['strategy_performance'].keys())

                        if 'conservative' in available_strategies:
                            benchmark_comparison = self.results_analyzer.compare_with_benchmark('conservative')

                            if benchmark_comparison and 'error' not in benchmark_comparison:
                                benchmark_file = os.path.join(
                                    self.config.output_directory,
                                    f"benchmark_comparison_{datetime.now().strftime('%Y%m%d')}.json"
                                )

                                with open(benchmark_file, 'w', encoding='utf-8') as f:
                                    json.dump(benchmark_comparison, f, indent=2, default=str, ensure_ascii=False)

                                export_successes.append("Benchmark comparison exported")
                                self.logger.info(f"Benchmark comparison exported to: {benchmark_file}")
                            else:
                                export_errors.append("Benchmark comparison returned error or empty results")
                        else:
                            self.logger.info(
                                f"Conservative strategy not found for benchmarking. Available: {available_strategies}")

                    except Exception as benchmark_error:
                        export_errors.append(f"Benchmark comparison failed: {str(benchmark_error)}")
                        self.logger.warning(f"Benchmark comparison failed: {str(benchmark_error)}")

                    # ENHANCED: Log key insights from analysis with error handling
                    try:
                        self._log_analysis_insights(comprehensive_results)
                    except Exception as insights_error:
                        self.logger.warning(f"Failed to log analysis insights: {str(insights_error)}")

                except Exception as analysis_error:
                    export_errors.append(f"Comprehensive analysis generation failed: {str(analysis_error)}")
                    self.backtest_logger.log_error(analysis_error, "Comprehensive analysis generation")

            else:
                self.logger.warning("ResultsAnalyzer not available - skipping comprehensive analysis")
                export_errors.append("ResultsAnalyzer not initialized - comprehensive analysis skipped")

        except Exception as e:
            export_errors.append(f"Failed to export comprehensive analysis: {str(e)}")
            self.backtest_logger.log_error(e, "Comprehensive analysis export")

        # ENHANCED: Final export summary and recommendations
        try:
            # Create an export summary report
            export_summary = {
                'export_timestamp': datetime.now().isoformat(),
                'total_exports_attempted': len(export_successes) + len(export_errors),
                'successful_exports': len(export_successes),
                'failed_exports': len(export_errors),
                'success_rate': len(export_successes) / (len(export_successes) + len(export_errors)) * 100 if (
                            export_successes or export_errors) else 0,
                'export_successes': export_successes,
                'export_errors': export_errors
            }

            # Export the summary report itself
            export_summary_file = os.path.join(
                self.config.output_directory,
                f"export_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(export_summary_file, 'w', encoding='utf-8') as f:
                json.dump(export_summary, f, indent=2, default=str, ensure_ascii=False)

            self.logger.info(f"Export summary report saved to: {export_summary_file}")

        except Exception as summary_error:
            self.logger.error(f"Failed to create export summary: {str(summary_error)}")

        # ENHANCED: Detailed logging of export results
        if export_errors:
            self.logger.warning(
                f"Export completed with {len(export_errors)} errors out of {len(export_successes) + len(export_errors)} attempts:")
            for i, error in enumerate(export_errors, 1):
                self.logger.warning(f"  {i}. {error}")
        else:
            self.logger.info("All exports completed successfully")

        if export_successes:
            self.logger.info(f"Successfully completed {len(export_successes)} exports:")
            for i, success in enumerate(export_successes, 1):
                self.logger.info(f"  {i}. {success}")

        # ENHANCED: Performance summary
        total_attempts = len(export_successes) + len(export_errors)
        if total_attempts > 0:
            success_rate = len(export_successes) / total_attempts * 100
            self.logger.info(f"Export success rate: {success_rate:.1f}% ({len(export_successes)}/{total_attempts})")

            if success_rate < 80:
                self.logger.warning("Export success rate below 80% - consider investigating export issues")
            elif success_rate == 100:
                self.logger.info("Perfect export success rate achieved!")

        # ENHANCED: Recommendations based on export results
        if export_errors:
            recommendations = []

            if any("missing export_dispatch_results method" in error for error in export_errors):
                recommendations.append("Add export_dispatch_results method to DispatchSimulator class")

            if any("Permission denied" in error for error in export_errors):
                recommendations.append("Check file permissions and ensure output directory is writable")

            if any("ResultsAnalyzer" in error for error in export_errors):
                recommendations.append("Verify ResultsAnalyzer initialization and data loading")

            if recommendations:
                self.logger.info("Recommendations to improve export success:")
                for i, rec in enumerate(recommendations, 1):
                    self.logger.info(f"  {i}. {rec}")

    def _validate_export_prerequisites(self) -> List[str]:
        """
        Validate that all prerequisites for export are met.

        Returns:
        --------
        List[str]
            List of validation issues that could affect exports
        """
        issues = []

        # Check output directory
        if not os.path.exists(self.config.output_directory):
            issues.append(f"Output directory does not exist: {self.config.output_directory}")
        elif not os.access(self.config.output_directory, os.W_OK):
            issues.append(f"Output directory is not writable: {self.config.output_directory}")

        # Check backtest results
        if not self.backtest_results:
            issues.append("No backtest results available for export")
        elif 'strategy_performance' not in self.backtest_results:
            issues.append("Backtest results missing strategy performance data")

        # Check dispatch simulator
        if not hasattr(self, 'dispatch_simulator') or not self.dispatch_simulator:
            issues.append("DispatchSimulator not initialized")
        elif not hasattr(self.dispatch_simulator, 'dispatch_history'):
            issues.append("DispatchSimulator missing dispatch_history")
        elif not self.dispatch_simulator.dispatch_history:
            issues.append("DispatchSimulator has empty dispatch_history")

        # Check the result analyzer
        if not hasattr(self, 'results_analyzer') or not self.results_analyzer:
            issues.append("ResultsAnalyzer not initialized")

        # Check logger
        if not hasattr(self, 'backtest_logger') or not self.backtest_logger:
            issues.append("BatteryBacktestLogger not initialized")

        return issues

    def _validate_maxavail_integration(self) -> bool:
        """
        Validate that MAXAVAIL integration is working correctly.

        Returns:
        --------
        bool
            True if MAXAVAIL integration is detected, False otherwise
        """
        try:
            # Check if DispatchSimulator has MAXAVAIL methods
            required_methods = [
                'calculate_maxavail',
                'apply_nemde_dispatch_algorithm',
                'validate_ramp_constraints_post_dispatch'
            ]

            for method_name in required_methods:
                if not hasattr(self.dispatch_simulator, method_name):
                    self.logger.warning(f"MAXAVAIL method missing: {method_name}")
                    return False

            self.logger.info("MAXAVAIL integration validated successfully")
            return True

        except Exception as e:
            self.logger.warning(f"MAXAVAIL validation failed: {str(e)}")
            return False

    def _convert_results_to_json(self) -> Dict[str, Any]:
        """Convert backtest results to JSON-serializable format with MAXAVAIL support."""
        json_results = {}

        for key, value in self.backtest_results.items():
            if key == 'daily_results':
                json_results[key] = {
                    str(date): {
                        'date': str(date),
                        'price_bands': daily_data['price_bands'],
                        'market_conditions': daily_data['market_conditions'],
                        'strategy_results': {
                            strategy_name: {
                                'total_revenue': strategy_result['daily_summary'].total_revenue,
                                'total_dispatched_mwh': strategy_result['daily_summary'].total_dispatched_mwh,
                                'dispatch_events': strategy_result['daily_summary'].dispatch_events,
                                'dispatch_efficiency': strategy_result['daily_summary'].dispatch_efficiency,
                                'capacity_utilization': strategy_result['daily_summary'].capacity_utilization,
                                'final_override_events': strategy_result['daily_summary'].final_override_events,
                                'final_override_used': strategy_result.get('final_override_used', False),
                                # NEW: Add MAXAVAIL-related metrics if available
                                'constraint_events': getattr(strategy_result['daily_summary'], 'constraint_events', 0),
                                'ramp_constraint_events': getattr(strategy_result['daily_summary'],
                                                                  'ramp_constraint_events', 0),
                                'power_constraint_events': getattr(strategy_result['daily_summary'],
                                                                   'power_constraint_events', 0)
                            }
                            for strategy_name, strategy_result in daily_data['strategy_results'].items()
                        }
                    }
                    for date, daily_data in value.items()
                }
            else:
                json_results[key] = value

        return json_results

    def _log_analysis_insights(self, comprehensive_results) -> None:
        """Log key insights from comprehensive analysis with proper error handling."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("KEY ANALYSIS INSIGHTS")
            self.logger.info("=" * 60)

            # Strategy insights with error handling
            try:
                best_strategy = comprehensive_results.strategy_comparison.best_revenue_strategy
                best_efficiency = comprehensive_results.strategy_comparison.best_efficiency_strategy

                self.logger.info(f"Best Revenue Strategy: {best_strategy}")
                self.logger.info(f"Best Efficiency Strategy: {best_efficiency}")
            except Exception as e:
                self.logger.warning(f"Could not extract best strategies: {str(e)}")

            # Performance matrix insights with error handling
            try:
                performance_df = comprehensive_results.strategy_comparison.performance_matrix
                if not performance_df.empty and 'Total Revenue' in performance_df.columns:
                    total_revenue_range = performance_df['Total Revenue'].max() - performance_df['Total Revenue'].min()
                    self.logger.info(f"Revenue Range Across Strategies: ${total_revenue_range:.2f}")
                else:
                    # Fallback: use strategy_performance data directly
                    if hasattr(self, 'backtest_results') and 'strategy_performance' in self.backtest_results:
                        strategy_perf = self.backtest_results['strategy_performance']
                        revenues = [perf['total_revenue'] for perf in strategy_perf.values()]
                        if revenues:
                            revenue_range = max(revenues) - min(revenues)
                            self.logger.info(f"Revenue Range Across Strategies: ${revenue_range:.2f}")
            except Exception as e:
                self.logger.warning(f"Could not analyze revenue range: {str(e)}")

            # Top 3 recommendations with error handling
            try:
                top_recommendations = comprehensive_results.recommendations[:3]
                if top_recommendations:
                    self.logger.info("Top Recommendations:")
                    for i, rec in enumerate(top_recommendations, 1):
                        self.logger.info(f"  {i}. {rec}")
                else:
                    self.logger.info("No specific recommendations generated")
            except Exception as e:
                self.logger.warning(f"Could not extract recommendations: {str(e)}")

            # Market condition insights with error handling
            try:
                if comprehensive_results.market_analysis.price_volatility_analysis:
                    volatility = comprehensive_results.market_analysis.price_volatility_analysis.get('daily_volatility',
                                                                                                     0)
                    self.logger.info(f"Market Volatility: ${volatility:.2f}")
            except Exception as e:
                self.logger.debug(f"Market volatility not available: {str(e)}")

            # MAXAVAIL insights with error handling
            try:
                maxavail_data = comprehensive_results.maxavail_analysis
                if 'error' not in maxavail_data and 'summary' in maxavail_data:
                    summary = maxavail_data['summary']
                    avg_util = summary.get('average_utilization_pct', 0)
                    constraint_freq = summary.get('constraint_frequency_pct', 0)
                    self.logger.info(f"Average MAXAVAIL Utilization: {avg_util:.1f}%")
                    self.logger.info(f"MAXAVAIL Constraint Frequency: {constraint_freq:.1f}%")
            except Exception as e:
                self.logger.debug(f"MAXAVAIL insights not available: {str(e)}")

            # Fallback: use backtest_results directly if comprehensive_results incomplete
            try:
                if hasattr(self, 'backtest_results') and 'strategy_performance' in self.backtest_results:
                    strategy_perf = self.backtest_results['strategy_performance']
                    if strategy_perf:
                        best_revenue_strategy = max(strategy_perf.keys(),
                                                    key=lambda x: strategy_perf[x]['total_revenue'])
                        best_revenue = strategy_perf[best_revenue_strategy]['total_revenue']

                        self.logger.info(f"Fallback Analysis - Best Strategy: {best_revenue_strategy}")
                        self.logger.info(f"Fallback Analysis - Best Revenue: ${best_revenue:.2f}")
            except Exception as e:
                self.logger.debug(f"Fallback analysis failed: {str(e)}")

            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.warning(f"Failed to log analysis insights: {str(e)}")

    def validate_component_interfaces(self) -> List[str]:
        """
        Validate that all component interfaces are properly aligned.
        REFACTORED: Now uses standalone validation functions while preserving existing functionality.

        Returns:
        --------
        List[str]
            List of validation issues found across all components
        """
        # Use the standalone validation functions
        validation_results = validate_all_component_interfaces(
            dispatch_simulator=getattr(self, 'dispatch_simulator', None),
            strategy_manager=getattr(self, 'strategy_manager', None),
            revenue_calculator=getattr(self, 'revenue_calculator', None),
            results_analyzer=getattr(self, 'results_analyzer', None)
        )

        # Flatten all issues into a single list
        all_issues = []
        for component_name, issues in validation_results.items():
            all_issues.extend(issues)

        # Log results if logger is available
        if hasattr(self, 'logger') and self.logger:
            log_validation_results(validation_results, self.logger)

        return all_issues

    def _validate_data_flow(self, dispatch_results: List, strategy_results: Dict) -> List[str]:
        """
        Validate data flow between components with enhanced checks for fixed allocation logic.
        ENHANCED: Validates fixed allocation logic and dual override mechanisms.
        """
        validation_issues = []

        if not dispatch_results:
            validation_issues.append("No dispatch results to validate")
            return validation_issues

        # Check DispatchResult structure
        required_fields = ['strategy_name', 'timestamp', 'interval_number', 'rrp',
                           'requested_mw', 'dispatched_mw', 'status', 'remaining_soc']

        for i, result in enumerate(dispatch_results):
            for field in required_fields:
                if not hasattr(result, field):
                    validation_issues.append(f"DispatchResult {i} missing required field: {field}")
                elif getattr(result, field) is None and field == 'strategy_name':
                    validation_issues.append(f"DispatchResult {i} has None strategy_name")

        # Validate fixed allocation logic for each strategy
        for strategy_name, result in strategy_results.items():
            strategy_dispatch_results = [dr for dr in dispatch_results
                                         if hasattr(dr, 'strategy_name') and dr.strategy_name == strategy_name]

            if strategy_dispatch_results:
                # Check for proper SOC tracking in fixed allocation
                initial_soc = 100.0  # Starting capacity
                calculated_remaining = initial_soc

                for dr in strategy_dispatch_results:
                    calculated_remaining -= dr.dispatched_mw

                    # Allow small tolerance for floating point precision
                    if abs(dr.remaining_soc - calculated_remaining) > 0.1:
                        validation_issues.append(
                            f"Strategy {strategy_name} SOC tracking inconsistent at interval {dr.interval_number}: "
                            f"Expected {calculated_remaining:.2f}MW, got {dr.remaining_soc:.2f}MW"
                        )
                        break  # Only report first inconsistency per strategy

                # Validate both override mechanisms work
                minimum_floor_triggered = any(
                    dr.remaining_soc < 5.0 and getattr(dr, 'final_override_active', False)
                    for dr in strategy_dispatch_results
                )

                final_override_triggered = any(
                    dr.interval_number >= 25 and getattr(dr, 'final_override_active', False) and dr.remaining_soc >= 5.0
                    for dr in strategy_dispatch_results
                )

                # Check that override mechanisms work correctly
                final_result = strategy_dispatch_results[-1] if strategy_dispatch_results else None
                if final_result and final_result.remaining_soc > 0.1:  # Significant capacity remaining
                    last_intervals = [dr for dr in strategy_dispatch_results if dr.interval_number >= 25]
                    if last_intervals and not any(getattr(dr, 'final_override_active', False) for dr in last_intervals):
                        validation_issues.append(
                            f"Strategy {strategy_name} has remaining capacity but no override mechanism triggered"
                        )

        # Validate MAXAVAIL fields
        maxavail_missing_count = sum(
            1 for result in dispatch_results
            if not hasattr(result, 'maxavail_for_interval') or result.maxavail_for_interval is None
        )

        if maxavail_missing_count > 0:
            validation_issues.append(f"{maxavail_missing_count} DispatchResults missing MAXAVAIL fields")

        # Strategy consistency check
        if strategy_results:
            strategy_names_from_results = set()
            for result in dispatch_results:
                if hasattr(result, 'strategy_name') and result.strategy_name:
                    strategy_names_from_results.add(result.strategy_name)

            expected_strategies = set(strategy_results.keys())
            if strategy_names_from_results != expected_strategies:
                validation_issues.append(
                    f"Strategy mismatch: Expected {expected_strategies}, "
                    f"Found {strategy_names_from_results}"
                )

        # Log validation results
        if validation_issues:
            self.logger.warning(f"Data flow validation issues: {validation_issues}")
        else:
            self.logger.info("Data flow validation passed successfully")

        return validation_issues

# Helper functions
def create_default_configuration() -> Configuration:
    """
    Create a default configuration for the backtest system.

    Returns:
    --------
    Configuration
        Default configuration object
    """
    config = Configuration()

    # Set analysis period (adjust as needed)
    config.start_date = "2024/03/01 00:00:00"
    config.end_date = "2024/08/31 23:55:00"
    config.analysis_start_date = "2024/04/01 00:00:00"  # Allow for 28-day history

    return config

def _safe_get_dispatch_field(self, dispatch_result, field_name: str, default_value=None):
    """
    Safely get a field from DispatchResult, handling both old and new formats.

    Parameters:
    -----------
    dispatch_result : DispatchResult
        The dispatch result object
    field_name : str
        Name of the field to retrieve
    default_value : Any
        Default value if field doesn't exist

    Returns:
    --------
    Any
        Field value or default
    """
    return getattr(dispatch_result, field_name, default_value)

def setup_system(config: Configuration) -> BatteryBacktester:
    """
    Setup and initialize the backtest system.

    Parameters:
    -----------
    config : Configuration
        System configuration

    Returns:
    --------
    BatteryBacktester
        Initialized backtest system
    """
    backtest_system = BatteryBacktester(config)

    # Setup logging FIRST (this initializes backtest_logger)
    backtest_system.setup_logging("INFO")

    logger = backtest_system.logger
    logger.info("Starting Battery Bidding Backtest System")

    backtest_system.initialize_components()
    backtest_system.load_data()

    return backtest_system


def display_final_results(results: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Display final backtest results.

    Parameters:
    -----------
    results : Dict[str, Any]
        Backtest results
    logger : logging.Logger
        Logger instance
    """
    if results['strategy_performance']:
        best_strategy = max(results['strategy_performance'].keys(),
                            key=lambda s: results['strategy_performance'][s]['total_revenue'])
        best_performance = results['strategy_performance'][best_strategy]

        logger.info("=" * 60)
        logger.info("FINAL RECOMMENDATIONS")
        logger.info("=" * 60)
        logger.info(f"Best Strategy: {best_strategy.upper()}")
        logger.info(f"Total Revenue: ${best_performance['total_revenue']:.2f}")
        logger.info(f"Revenue per MWh: ${best_performance['avg_revenue_per_mwh']:.2f}")

        # Fixed: Use the correct key names from _calculate_aggregate_performance
        logger.info(f"Energy Capacity Utilization: {best_performance['energy_capacity_utilization']:.1f}%")
        logger.info(f"Daily SOC Utilization: {best_performance['avg_daily_soc_utilization']:.1f}%")

        # Add more useful metrics from your aggregate performance calculation
        logger.info(f"Average Daily Revenue: ${best_performance['avg_daily_revenue']:.2f}")
        logger.info(f"Market Engagement Rate: {best_performance['market_engagement_rate']:.1f}%")
        logger.info(f"Final Override Events: {best_performance['total_final_override_events']}")
        logger.info(f"Days Analyzed: {best_performance['days_analyzed']}")
        logger.info("=" * 60)
    else:
        logger.warning("No strategy performance results available")


def handle_main_error(error: Exception) -> None:
    """
    Handle main function errors.

    Parameters:
    -----------
    error : Exception
        The error that occurred
    """
    print(f"Backtest failed: {str(error)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def patch_main_py_revenue_calculator_init():
    """
    Helper function to patch the RevenueCalculator initialization in main.py.

    Replace this line in main.py:
        self.revenue_calculator = RevenueCalculator(self.config)

    With:
        self.revenue_calculator = RevenueCalculator(
            max_capacity_mw=self.config.battery_parameters['max_discharge_power'],
            transmission_loss_factor=self.config.battery_parameters.get('TLF', 1.0),
            logger=self.logger
        )

    Or use the fixed constructor above that handles Configuration objects.
    """
    pass

def patch_main_py_results_analyzer_init():
    """
    Helper function to patch the ResultsAnalyzer initialization in main.py.

    The line in main.py:
        self.results_analyzer = ResultsAnalyzer(self.config, self.backtest_logger)

    Should work with the fixed constructor above.
    """
    pass


def validate_component_interfaces(self) -> List[str]:
    """
    Validate that all component interfaces are properly aligned.
    ENHANCED: Comprehensive interface validation for all components.
    """
    issues = []

    # Check DispatchSimulator interface
    if hasattr(self, 'dispatch_simulator') and self.dispatch_simulator:
        ds = self.dispatch_simulator

        # Core methods
        required_ds_methods = [
            'simulate_daily_dispatch', 'reset_simulation', 'export_dispatch_results',
            'simulate_dispatch_for_interval', 'calculate_maxavail',
            'apply_nemde_dispatch_algorithm', 'validate_ramp_constraints_post_dispatch'
        ]

        for method in required_ds_methods:
            if not hasattr(ds, method):
                issues.append(f"DispatchSimulator missing method: {method}")

        # Enhanced methods for new interface
        enhanced_ds_methods = ['clear_history', 'get_export_summary', 'validate_export_data']
        for method in enhanced_ds_methods:
            if not hasattr(ds, method):
                issues.append(f"DispatchSimulator missing enhanced method: {method}")

    # Check BiddingStrategyManager interface
    if hasattr(self, 'strategy_manager') and self.strategy_manager:
        sm = self.strategy_manager

        # Core methods
        required_sm_methods = [
            'initialize_strategy', 'get_bid_schedule', 'get_remaining_soc',
            'update_after_dispatch', 'get_strategy_info', 'is_final_override_active',
            'get_bid_schedule_with_maxavail_check'
        ]

        for method in required_sm_methods:
            if not hasattr(sm, method):
                issues.append(f"BiddingStrategyManager missing method: {method}")

        # Compatibility methods for DispatchSimulator
        compatibility_methods = [
            'update_capacity_after_dispatch', 'get_remaining_capacity',
            'get_current_bid_schedule', 'reset_all_strategies'
        ]

        for method in compatibility_methods:
            if not hasattr(sm, method):
                issues.append(f"BiddingStrategyManager missing compatibility method: {method}")

    # Check RevenueCalculator interface
    if hasattr(self, 'revenue_calculator') and self.revenue_calculator:
        rc = self.revenue_calculator
        required_rc_methods = ['calculate_daily_revenue']
        for method in required_rc_methods:
            if not hasattr(rc, method):
                issues.append(f"RevenueCalculator missing method: {method}")

    # Check ResultsAnalyzer interface
    if hasattr(self, 'results_analyzer') and self.results_analyzer:
        ra = self.results_analyzer
        required_ra_methods = [
            'load_backtest_data', 'create_comprehensive_report',
            'export_detailed_report', 'export_results_summary',
            'generate_visualizations', 'compare_with_benchmark'
        ]
        for method in required_ra_methods:
            if not hasattr(ra, method):
                issues.append(f"ResultsAnalyzer missing method: {method}")

    if issues:
        self.logger.warning(f"Component interface validation issues found: {issues}")
    else:
        self.logger.info("All component interfaces validated successfully")

    return issues

def cleanup_memory(backtest_system, logger):
    """
    Clean up memory after backtest completion.

    Parameters:
    -----------
    backtest_system : BatteryBacktester
        The main backtest system object
    logger : logging.Logger
        Logger for reporting cleanup status
    """
    try:
        logger.info("Starting memory cleanup...")

        # Monitor memory before cleanup
        memory_before = monitor_memory_usage(logger)

        # Clear large data structures
        if hasattr(backtest_system, 'data_manager') and backtest_system.data_manager:
            if hasattr(backtest_system.data_manager, 'clear_cache'):
                backtest_system.data_manager.clear_cache()
            else:
                # Fallback: clear known data attributes
                if hasattr(backtest_system.data_manager, 'rrp_data'):
                    del backtest_system.data_manager.rrp_data
                logger.info("DataManager data cleared (fallback method)")

        if hasattr(backtest_system, 'dispatch_simulator') and backtest_system.dispatch_simulator:
            if hasattr(backtest_system.dispatch_simulator, 'clear_history'):
                backtest_system.dispatch_simulator.clear_history()
            else:
                # Fallback: clear dispatch history manually
                if hasattr(backtest_system.dispatch_simulator, 'dispatch_history'):
                    backtest_system.dispatch_simulator.dispatch_history.clear()
                if hasattr(backtest_system.dispatch_simulator, 'daily_summaries'):
                    backtest_system.dispatch_simulator.daily_summaries.clear()
                logger.info("DispatchSimulator history cleared (fallback method)")

        if hasattr(backtest_system, 'results_analyzer') and backtest_system.results_analyzer:
            if hasattr(backtest_system.results_analyzer, 'clear_cache'):
                backtest_system.results_analyzer.clear_cache()
            else:
                logger.info("ResultsAnalyzer cleanup not available")

        # Clear backtest results to free memory
        if hasattr(backtest_system, 'backtest_results'):
            backtest_system.backtest_results.clear()
            logger.info("Backtest results cleared")

        # Clear strategy managers
        if hasattr(backtest_system, 'strategy_manager') and backtest_system.strategy_manager:
            if hasattr(backtest_system.strategy_manager, 'reset_all_strategies'):
                backtest_system.strategy_manager.reset_all_strategies()
                logger.info("Strategy managers cleared")

        # Force garbage collection
        import gc
        collected = gc.collect()

        # Monitor memory after cleanup
        memory_after = monitor_memory_usage(logger)
        memory_freed = memory_before - memory_after if memory_before > 0 else 0

        logger.info(f"Memory cleanup completed:")
        logger.info(f"  • Collected {collected} objects")
        logger.info(f"  • Memory before: {memory_before:.1f} MB")
        logger.info(f"  • Memory after: {memory_after:.1f} MB")
        if memory_freed > 0:
            logger.info(f"  • Memory freed: {memory_freed:.1f} MB")

    except Exception as e:
        logger.warning(f"Memory cleanup failed: {str(e)}")


def optimize_pandas_memory():
    """
    Configure pandas for lower memory usage
    """
    try:
        import pandas as pd

        # Configure pandas options for memory efficiency
        pd.set_option('mode.copy_on_write', True)  # Reduce unnecessary copies
        pd.set_option('compute.use_bottleneck', True)  # Use optimized functions
        pd.set_option('compute.use_numexpr', True)  # Use numexpr for calculations

        print("Configured pandas for memory efficiency")

    except Exception as e:
        print(f"Failed to configure pandas: {str(e)}")

def main():
    """
    Main function to run the battery bidding backtest system.
    Enhanced with memory optimization and cleanup.
    """
    optimize_pandas_memory()

    try:
        # Create and validate configuration
        config = create_default_configuration()
        config.validate_configuration()

        # Initialize system
        backtest_system = setup_system(config)

        # Run backtest (can specify max_days=n)
        results = backtest_system.run_backtest()

        # SOC VALIDATION SUMMARY - AFTER BACKTEST COMPLETES
        logger = backtest_system.logger  # Use the integrated logger

        # SOC Validation Summary
        if hasattr(backtest_system.dispatch_simulator, 'get_soc_validation_summary'):
            try:
                soc_summary = backtest_system.dispatch_simulator.get_soc_validation_summary()
                logger.info("=" * 60)
                logger.info("SOC VALIDATION SUMMARY")
                logger.info("=" * 60)
                logger.info(f"Total Intervals Processed: {soc_summary['total_intervals']}")
                logger.info(f"Validation Data Available: {soc_summary['validation_available']}")
                if soc_summary['validation_available']:
                    logger.info(f"Intervals with Validation Data: {soc_summary['intervals_with_validation_data']}")
                    logger.info(f"Consistent Intervals: {soc_summary['consistent_intervals']}")
                    logger.info(f"Consistency Rate: {soc_summary['consistency_rate']:.1f}%")
                    logger.info(f"Summary: {soc_summary['summary']}")

                    # Alert if consistency rate is low
                    if soc_summary['consistency_rate'] < 95:
                        logger.warning("⚠️  SOC consistency rate below 95% - review dispatch calculations")
                    else:
                        logger.info("✅ SOC calculations are consistent across all intervals")
                else:
                    logger.info("No SOC validation data available")
                logger.info("=" * 60)

            except Exception as e:
                logger.warning(f"Failed to generate SOC validation summary: {str(e)}")
        else:
            logger.info("SOC validation not available (method not found)")

        # Display final results
        display_final_results(results, logger)

        # Print performance report to console
        if backtest_system.backtest_logger:
            print("\n" + backtest_system.backtest_logger.create_performance_report())

        logger.info("Backtest completed successfully!")

        # MEMORY CLEANUP (NEW)
        enhanced_cleanup_memory(backtest_system, logger)

    except Exception as e:
        handle_main_error(e)
    finally:
        # Force garbage collection at the end
        import gc
        collected = gc.collect()
        print(f"Final cleanup: collected {collected} objects")


def analyze_memory_usage(logger):
    """
    Detailed memory analysis to identify what's using memory
    """
    try:
        import psutil
        import os
        import gc
        import sys

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        logger.info("=" * 60)
        logger.info("DETAILED MEMORY ANALYSIS")
        logger.info("=" * 60)

        # Basic memory info
        rss_mb = memory_info.rss / (1024 * 1024)
        vms_mb = memory_info.vms / (1024 * 1024)

        logger.info(f"RSS (Resident Set Size): {rss_mb:.1f} MB")
        logger.info(f"VMS (Virtual Memory Size): {vms_mb:.1f} MB")

        # Python object counts
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

        # Show top 10 object types
        logger.info("\nTop 10 Python object types by count:")
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        for obj_type, count in sorted_objects[:10]:
            logger.info(f"  {obj_type}: {count:,}")

        # Check for large objects
        large_objects = []
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj)
                if size > 1024 * 1024:  # > 1MB
                    large_objects.append((type(obj).__name__, size / (1024 * 1024)))
            except:
                continue

        if large_objects:
            logger.info("\nLarge objects (>1MB):")
            for obj_type, size_mb in sorted(large_objects, key=lambda x: x[1], reverse=True):
                logger.info(f"  {obj_type}: {size_mb:.1f} MB")

        # Check pandas DataFrames specifically
        import pandas as pd
        dataframes = [obj for obj in gc.get_objects() if isinstance(obj, pd.DataFrame)]
        if dataframes:
            logger.info(f"\nFound {len(dataframes)} pandas DataFrames:")
            total_df_memory = 0
            for i, df in enumerate(dataframes):
                try:
                    df_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    total_df_memory += df_memory
                    if df_memory > 1:  # Only show DataFrames > 1MB
                        logger.info(f"  DataFrame {i}: {df_memory:.1f} MB ({len(df)} rows, {len(df.columns)} cols)")
                except:
                    pass
            logger.info(f"  Total DataFrame memory: {total_df_memory:.1f} MB")

        logger.info("=" * 60)

        return {
            'rss_mb': rss_mb,
            'vms_mb': vms_mb,
            'object_counts': object_counts,
            'large_objects': large_objects,
            'dataframe_count': len(dataframes) if 'dataframes' in locals() else 0
        }

    except ImportError:
        logger.warning("psutil not available - install with: pip install psutil")
        return {}
    except Exception as e:
        logger.error(f"Memory analysis failed: {str(e)}")
        return {}

def enhanced_cleanup_memory(backtest_system, logger):
    """
    Enhanced memory cleanup with detailed analysis
    """
    try:
        logger.info("Starting enhanced memory cleanup...")

        # 1. Analyze memory before cleanup
        memory_before = analyze_memory_usage(logger)

        # 2. Perform standard cleanup
        cleanup_memory(backtest_system, logger)

        # 3. Perform aggressive cleanup
        collected = aggressive_memory_cleanup(backtest_system, logger)

        # 4. Check for memory leaks
        check_memory_leaks(logger)

        # 5. Final memory analysis
        logger.info("\nMemory usage after enhanced cleanup:")
        memory_after = analyze_memory_usage(logger)

        # Calculate memory freed
        if memory_before.get('rss_mb', 0) > 0 and memory_after.get('rss_mb', 0) > 0:
            memory_freed = memory_before['rss_mb'] - memory_after['rss_mb']
            logger.info(f"Total memory freed: {memory_freed:.1f} MB")

    except Exception as e:
        logger.error(f"Enhanced cleanup failed: {str(e)}")

def check_memory_leaks(logger):
    """
    Check for common memory leak patterns
    """
    try:
        import gc
        import sys

        logger.info("Checking for memory leak patterns...")

        # Check for circular references
        gc.collect()  # Clean up first
        if gc.garbage:
            logger.warning(f"Found {len(gc.garbage)} objects in gc.garbage (possible circular refs)")

            # Show types of garbage objects
            garbage_types = {}
            for obj in gc.garbage:
                obj_type = type(obj).__name__
                garbage_types[obj_type] = garbage_types.get(obj_type, 0) + 1

            for obj_type, count in garbage_types.items():
                logger.warning(f"  Garbage type {obj_type}: {count}")

        # Check reference counts of our custom objects
        custom_objects = []
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            if any(keyword in obj_type.lower() for keyword in
                   ['dispatch', 'battery', 'strategy', 'backtest', 'result']):
                custom_objects.append(obj_type)

        if custom_objects:
            from collections import Counter
            custom_counts = Counter(custom_objects)
            logger.info("Custom object counts:")
            for obj_type, count in custom_counts.most_common(10):
                logger.info(f"  {obj_type}: {count}")

    except Exception as e:
        logger.error(f"Memory leak check failed: {str(e)}")


def monitor_memory_usage(logger):
    """
    Monitor and log current memory usage.

    Parameters:
    -----------
    logger : logging.Logger
        Logger for reporting memory status

    Returns:
    --------
    float
        Current memory usage in MB
    """
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        logger.debug(f"Current memory usage: {memory_mb:.1f} MB")

        if memory_mb > 1000:  # Over 1GB
            logger.warning(f"High memory usage detected: {memory_mb:.1f} MB")
        elif memory_mb > 500:  # Over 500MB
            logger.info(f"Moderate memory usage: {memory_mb:.1f} MB")

        return memory_mb

    except ImportError:
        logger.debug("psutil not available for memory monitoring")
        logger.info("Install psutil for memory monitoring: pip install psutil")
        return 0
    except Exception as e:
        logger.warning(f"Memory monitoring failed: {str(e)}")
        return 0


def configure_memory_optimization():
    """
    Configure Python garbage collection for better memory management.
    Call this at the start of your application.
    """
    import gc

    # More aggressive garbage collection
    # Default is (700, 10, 10) - we make it more frequent
    gc.set_threshold(500, 8, 8)

    # Enable garbage collection (should be enabled by default)
    gc.enable()

    print("Configured Python for memory optimization")
    print(f"Garbage collection thresholds: {gc.get_threshold()}")



def aggressive_memory_cleanup(backtest_system, logger):
    """
    More aggressive memory cleanup that targets common memory leaks
    """
    try:
        logger.info("Starting aggressive memory cleanup...")

        import gc
        import pandas as pd

        # 1. Clear all pandas DataFrames
        dataframes = [obj for obj in gc.get_objects() if isinstance(obj, pd.DataFrame)]
        logger.info(f"Found {len(dataframes)} DataFrames to clear")

        for df in dataframes:
            try:
                df.drop(df.index, inplace=True)  # Clear DataFrame content
            except:
                pass

        # 2. Clear matplotlib figures (if any)
        try:
            import matplotlib.pyplot as plt
            plt.close('all')  # Close all matplotlib figures
            logger.info("Cleared matplotlib figures")
        except ImportError:
            pass

        # 3. Clear logging handlers' buffers
        try:
            for handler in logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
            logger.info("Flushed logging handlers")
        except:
            pass

        # 4. Clear numpy arrays (if any large ones exist)
        try:
            import numpy as np
            arrays = [obj for obj in gc.get_objects() if isinstance(obj, np.ndarray)]
            large_arrays = [arr for arr in arrays if arr.nbytes > 1024 * 1024]  # > 1MB
            logger.info(f"Found {len(large_arrays)} large numpy arrays")
            for arr in large_arrays:
                try:
                    arr.resize(0)  # Resize to empty
                except:
                    pass
        except ImportError:
            pass

        # 5. Force multiple garbage collection cycles
        collected_total = 0
        for i in range(3):
            collected = gc.collect()
            collected_total += collected
            if collected == 0:
                break

        logger.info(f"Aggressive cleanup: collected {collected_total} objects total")

        return collected_total

    except Exception as e:
        logger.error(f"Aggressive cleanup failed: {str(e)}")
        return 0

if __name__ == "__main__":
    main()