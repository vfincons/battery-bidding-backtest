import pandas as pd
import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Try to import matplotlib, but handle if not available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class BacktestPerformanceMetrics:
    """Comprehensive backtest performance metrics with MAXAVAIL analysis"""
    # Basic performance
    total_revenue: float
    total_dispatched_mwh: float
    average_revenue_per_mwh: float
    capacity_utilization: float
    dispatch_frequency: float

    # MAXAVAIL-specific metrics
    maxavail_constraint_frequency: float
    average_maxavail_utilization: float
    maxavail_revenue_impact: float
    nemde_dispatch_efficiency: float

    # Constraint analysis
    constraint_type_distribution: Dict[str, int]
    most_common_constraint: str
    constraint_revenue_loss: float

    # Strategy performance
    strategy_name: str
    simulation_period: str
    total_intervals: int
    successful_dispatch_intervals: int


@dataclass
class MAXAVAILAnalysisResults:
    """Detailed MAXAVAIL constraint analysis results"""
    # Utilization metrics
    average_utilization: float
    median_utilization: float
    utilization_std: float
    utilization_percentiles: Dict[str, float]

    # Constraint frequency
    total_intervals: int
    maxavail_constrained_intervals: int
    constraint_frequency_pct: float

    # Revenue impact
    total_revenue_loss: float
    average_revenue_loss_per_event: float
    revenue_impact_pct: float

    # Efficiency metrics
    average_nemde_efficiency: float
    capacity_withheld_mw: float
    dispatch_success_rate: float

    # Temporal patterns
    constraint_by_hour: Dict[int, int]
    constraint_by_interval: Dict[int, int]


@dataclass
class ConstraintAnalysisResults:
    """Comprehensive constraint type analysis"""
    constraint_counts: Dict[str, int]
    constraint_frequencies: Dict[str, float]
    constraint_revenue_impacts: Dict[str, float]
    constraint_efficiency_impacts: Dict[str, float]
    most_limiting_constraint: str
    total_constraint_events: int
    unconstrained_intervals: int


class ResultsAnalyzer:
    """
    Enhanced results analyzer with comprehensive MAXAVAIL constraint analysis.

    Analyzes backtest results including dispatch efficiency, constraint impacts,
    MAXAVAIL utilization patterns, and revenue optimization opportunities.
    Integrates with DispatchSimulator and RevenueCalculator outputs.
    """

    def __init__(self, config, backtest_logger=None):
        """
        Initialize results analyzer.

        Parameters:
        -----------
        config : Configuration
            Configuration object containing battery parameters and settings
        backtest_logger : BatteryBacktestLogger, optional
            Advanced battery backtest logger for enhanced logging capabilities
        """
        # Store configuration and setup logging
        self.config = config
        self.backtest_logger = backtest_logger

        # Setup logger
        if backtest_logger and hasattr(backtest_logger, 'logger'):
            self.logger = backtest_logger.logger
        else:
            self.logger = logging.getLogger(__name__)

        # Data storage for backtest results
        self.backtest_results = {}
        self.dispatch_history = []
        self.revenue_breakdown_history = []
        self.strategy_performance = {}

        # MAXAVAIL analysis data
        self.maxavail_events = []
        self.constraint_analysis: Optional[ConstraintAnalysisResults] = None
        self.utilization_analysis: Optional[MAXAVAILAnalysisResults] = None

        # Visualization settings
        self.figure_size = (12, 8)
        self.color_palette = "viridis"

        self.logger.info("ResultsAnalyzer initialized with MAXAVAIL analysis capabilities")

    def load_backtest_data(self, backtest_results: Dict[str, Any],
                           dispatch_history: List,
                           revenue_breakdown_history: Optional[List] = None) -> None:
        """Load backtest data with proper strategy attribution"""
        self.logger.info("Loading backtest data with MAXAVAIL analysis...")

        # Store core data
        self.backtest_results = backtest_results
        self.dispatch_history = dispatch_history
        self.revenue_breakdown_history = revenue_breakdown_history or []

        # Ensure strategy names are properly set
        self._ensure_strategy_attribution()

        # Validate MAXAVAIL fields in dispatch history
        self._validate_maxavail_fields()

        # Extract MAXAVAIL events for analysis
        self._extract_maxavail_events()

        # Perform constraint analysis
        self._analyze_constraints()

        # Calculate utilization metrics
        self._analyze_maxavail_utilization()

        # Generate strategy performance metrics
        self._calculate_strategy_performance()

        self.logger.info(f"Loaded {len(self.dispatch_history)} dispatch intervals for analysis")
        self.logger.info(f"Found {len(self.maxavail_events)} MAXAVAIL constraint events")

    def _validate_maxavail_fields(self) -> None:
        """Validate that dispatch history contains required MAXAVAIL fields"""
        if not self.dispatch_history:
            self.logger.warning("No dispatch history provided")
            return

        # Check for MAXAVAIL fields in first result
        sample_result = self.dispatch_history[0]
        required_fields = [
            'maxavail_for_interval',
            'maxavail_utilization',
            'total_bandavail',
            'maxavail_constraint_applied'
        ]

        missing_fields = []
        for field in required_fields:
            if not hasattr(sample_result, field):
                missing_fields.append(field)

        if missing_fields:
            self.logger.warning(f"Missing MAXAVAIL fields in DispatchResult: {missing_fields}")
        else:
            self.logger.info("All required MAXAVAIL fields present in dispatch history")

    def _extract_maxavail_events(self) -> None:
        """Extract and categorize MAXAVAIL constraint events"""
        self.maxavail_events = []

        for result in self.dispatch_history:
            # Extract MAXAVAIL-specific data
            maxavail_event = {
                'timestamp': result.timestamp,
                'interval_number': result.interval_number,
                'rrp': result.rrp,
                'maxavail': getattr(result, 'maxavail_for_interval', 0.0),
                'total_bandavail': getattr(result, 'total_bandavail', 0.0),
                'requested_mw': result.requested_mw,
                'dispatched_mw': result.dispatched_mw,
                'maxavail_utilization': getattr(result, 'maxavail_utilization', 0.0),
                'maxavail_constraint_applied': getattr(result, 'maxavail_constraint_applied', False),
                'constraint_applied': result.constraint_applied,
                'dispatch_efficiency': result.get_dispatch_efficiency(),
                'revenue': getattr(result, 'revenue', 0.0),
                'status': result.status.value if hasattr(result.status, 'value') else str(result.status)
            }

            self.maxavail_events.append(maxavail_event)

    def _analyze_constraints(self) -> None:
        """Analyze constraint type distribution and impacts"""
        if not self.dispatch_history:
            return

        # Count constraint types
        constraint_counts = {}
        constraint_revenue_impacts = {}
        constraint_efficiency_impacts = {}

        total_revenue_loss = 0.0
        total_efficiency_loss = 0.0

        for result in self.dispatch_history:
            constraint = result.constraint_applied or "none"

            # Count occurrences
            constraint_counts[constraint] = constraint_counts.get(constraint, 0) + 1

            # Calculate revenue impact (simplified - actual impact would come from RevenueCalculator)
            if constraint != "none" and result.requested_mw > 0:
                revenue_loss = (result.requested_mw - result.dispatched_mw) * (result.rrp / 12)
                total_revenue_loss += revenue_loss
                constraint_revenue_impacts[constraint] = constraint_revenue_impacts.get(constraint, 0) + revenue_loss

                efficiency_loss = 1.0 - result.get_dispatch_efficiency()
                total_efficiency_loss += efficiency_loss
                constraint_efficiency_impacts[constraint] = constraint_efficiency_impacts.get(constraint,
                                                                                              0) + efficiency_loss

        # Calculate frequencies
        total_intervals = len(self.dispatch_history)
        constraint_frequencies = {
            constraint: (count / total_intervals * 100)
            for constraint, count in constraint_counts.items()
        }

        # Find most limiting constraint (excluding "none")
        limiting_constraints = {k: v for k, v in constraint_counts.items() if k != "none"}
        most_limiting_constraint = max(limiting_constraints.keys(),
                                       key=lambda x: limiting_constraints[x]) if limiting_constraints else "none"

        self.constraint_analysis = ConstraintAnalysisResults(
            constraint_counts=constraint_counts,
            constraint_frequencies=constraint_frequencies,
            constraint_revenue_impacts=constraint_revenue_impacts,
            constraint_efficiency_impacts=constraint_efficiency_impacts,
            most_limiting_constraint=most_limiting_constraint,
            total_constraint_events=sum(v for k, v in constraint_counts.items() if k != "none"),
            unconstrained_intervals=constraint_counts.get("none", 0)
        )

        self.logger.info(
            f"Constraint analysis complete: {self.constraint_analysis.total_constraint_events} constraint events")
        self.logger.info(f"Most limiting constraint: {most_limiting_constraint}")

    def _analyze_maxavail_utilization(self) -> None:
        """Analyze MAXAVAIL utilization patterns and efficiency"""
        if not self.maxavail_events:
            return

        # Extract utilization data
        utilization_data = [event['maxavail_utilization'] for event in self.maxavail_events
                            if event['maxavail_utilization'] is not None]

        if not utilization_data:
            self.logger.warning("No MAXAVAIL utilization data found")
            return

        # Calculate utilization statistics
        utilization_array = np.array(utilization_data)
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        utilization_percentiles: Dict[str, float] = {
            f"p{p}": float(np.percentile(utilization_array, p)) for p in percentiles
        }

        # Count MAXAVAIL constraint events
        maxavail_constrained = sum(1 for event in self.maxavail_events
                                   if event['maxavail_constraint_applied'])

        # Calculate revenue impacts
        total_revenue_loss = 0.0
        maxavail_revenue_loss = 0.0

        for event in self.maxavail_events:
            if event['constraint_applied'] and event['constraint_applied'] != "none":
                revenue_loss = (event['requested_mw'] - event['dispatched_mw']) * (event['rrp'] / 12)
                total_revenue_loss += revenue_loss

                if event['constraint_applied'] == 'maxavail':
                    maxavail_revenue_loss += revenue_loss

        # Calculate efficiency metrics
        nemde_efficiencies = [event['dispatch_efficiency'] for event in self.maxavail_events]
        average_nemde_efficiency = np.mean(nemde_efficiencies) if nemde_efficiencies else 0.0

        # Calculate capacity metrics
        total_capacity_withheld = sum(event['requested_mw'] - event['dispatched_mw']
                                      for event in self.maxavail_events)

        dispatch_success_events = sum(1 for event in self.maxavail_events if event['dispatched_mw'] > 0)
        dispatch_success_rate = dispatch_success_events / len(self.maxavail_events) * 100

        # Temporal pattern analysis
        constraint_by_hour = {}
        constraint_by_interval = {}

        for event in self.maxavail_events:
            if event['maxavail_constraint_applied']:
                hour = event['timestamp'].hour
                interval = event['interval_number']

                constraint_by_hour[hour] = constraint_by_hour.get(hour, 0) + 1
                constraint_by_interval[interval] = constraint_by_interval.get(interval, 0) + 1

        self.utilization_analysis = MAXAVAILAnalysisResults(
            average_utilization=float(np.mean(utilization_array)),
            median_utilization=float(np.median(utilization_array)),
            utilization_std=float(np.std(utilization_array)),
            utilization_percentiles=utilization_percentiles,
            total_intervals=len(self.maxavail_events),
            maxavail_constrained_intervals=maxavail_constrained,
            constraint_frequency_pct=float(maxavail_constrained / len(self.maxavail_events) * 100),
            total_revenue_loss=float(maxavail_revenue_loss),
            average_revenue_loss_per_event=float(maxavail_revenue_loss / max(1, maxavail_constrained)),
            revenue_impact_pct=float(maxavail_revenue_loss / max(1, total_revenue_loss) * 100),
            average_nemde_efficiency=float(average_nemde_efficiency),
            capacity_withheld_mw=float(total_capacity_withheld),
            dispatch_success_rate=float(dispatch_success_rate),
            constraint_by_hour=constraint_by_hour,
            constraint_by_interval=constraint_by_interval
        )

        self.logger.info(f"MAXAVAIL utilization analysis complete:")
        self.logger.info(f"  Average utilization: {self.utilization_analysis.average_utilization:.1%}")
        self.logger.info(f"  MAXAVAIL constraint frequency: {self.utilization_analysis.constraint_frequency_pct:.1f}%")
        self.logger.info(f"  NEMDE efficiency: {self.utilization_analysis.average_nemde_efficiency:.1%}")

    def _calculate_strategy_performance(self) -> None:
        """Calculate comprehensive strategy performance metrics with proper data access"""
        if not self.dispatch_history:
            return

        # Group by strategy if available, otherwise treat as single strategy
        strategies = {}
        for result in self.dispatch_history:
            strategy_name = getattr(result, 'strategy_name', 'default_strategy')

            if strategy_name not in strategies:
                strategies[strategy_name] = []
            strategies[strategy_name].append(result)

        # Add debugging
        self.logger.debug(f"Found strategies: {list(strategies.keys())}")

        for strategy_name, results in strategies.items():
            self.logger.debug(f"Processing strategy {strategy_name} with {len(results)} results")

            # Basic metrics
            total_revenue = sum(getattr(r, 'revenue', 0.0) for r in results)
            total_dispatched_mwh = sum(r.dispatched_mw / 12 for r in results)  # Convert MW to MWh
            successful_dispatches = sum(1 for r in results if r.dispatched_mw > 0)

            # MAXAVAIL metrics
            maxavail_constraints = sum(1 for r in results
                                       if getattr(r, 'maxavail_constraint_applied', False))
            maxavail_utilizations = [getattr(r, 'maxavail_utilization', 0.0) for r in results
                                     if getattr(r, 'maxavail_utilization', None) is not None]

            # Constraint analysis
            constraint_counts = {}
            for r in results:
                constraint = r.constraint_applied or "none"
                constraint_counts[constraint] = constraint_counts.get(constraint, 0) + 1

            most_common_constraint = max(constraint_counts.keys(),
                                         key=lambda x: constraint_counts[x]) if constraint_counts else "none"

            # Calculate performance metrics with proper float conversion
            performance = BacktestPerformanceMetrics(
                total_revenue=float(total_revenue),
                total_dispatched_mwh=float(total_dispatched_mwh),
                average_revenue_per_mwh=float(total_revenue / max(1, total_dispatched_mwh)),
                capacity_utilization=float(total_dispatched_mwh / len(results) * 12),  # Approximate
                dispatch_frequency=float(successful_dispatches / len(results) * 100),
                maxavail_constraint_frequency=float(maxavail_constraints / len(results) * 100),
                average_maxavail_utilization=float(np.mean(maxavail_utilizations)) if maxavail_utilizations else 0.0,
                maxavail_revenue_impact=0.0,  # Would be calculated from RevenueCalculator
                nemde_dispatch_efficiency=float(np.mean([r.get_dispatch_efficiency() for r in results])),
                constraint_type_distribution=constraint_counts,
                most_common_constraint=most_common_constraint,
                constraint_revenue_loss=0.0,  # Would be calculated from RevenueCalculator
                strategy_name=strategy_name,
                simulation_period=f"{results[0].timestamp.date()} to {results[-1].timestamp.date()}",
                total_intervals=len(results),
                successful_dispatch_intervals=successful_dispatches
            )

            self.strategy_performance[strategy_name] = performance

            # Log performance for debugging
            self.logger.debug(f"Strategy {strategy_name} performance: "
                              f"Revenue={performance.total_revenue:.2f}, "
                              f"Efficiency={performance.nemde_dispatch_efficiency:.3f}")

        self.logger.info(f"Strategy performance calculated for {len(strategies)} strategies")

    def generate_maxavail_utilization_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive MAXAVAIL utilization report.

        Returns:
        --------
        Dict[str, Any]
            Detailed MAXAVAIL utilization analysis
        """
        if not self.utilization_analysis:
            return {'error': 'No MAXAVAIL utilization analysis available'}

        return {
            'summary': {
                'average_utilization_pct': self.utilization_analysis.average_utilization * 100,
                'median_utilization_pct': self.utilization_analysis.median_utilization * 100,
                'utilization_variability': self.utilization_analysis.utilization_std,
                'constraint_frequency_pct': self.utilization_analysis.constraint_frequency_pct,
                'nemde_efficiency_pct': self.utilization_analysis.average_nemde_efficiency * 100
            },
            'utilization_distribution': {
                f"{k}_pct": v * 100 for k, v in self.utilization_analysis.utilization_percentiles.items()
            },
            'constraint_patterns': {
                'by_hour': self.utilization_analysis.constraint_by_hour,
                'by_interval': self.utilization_analysis.constraint_by_interval
            },
            'revenue_impact': {
                'total_revenue_loss': self.utilization_analysis.total_revenue_loss,
                'average_loss_per_event': self.utilization_analysis.average_revenue_loss_per_event,
                'revenue_impact_pct': self.utilization_analysis.revenue_impact_pct
            },
            'efficiency_metrics': {
                'dispatch_success_rate_pct': self.utilization_analysis.dispatch_success_rate,
                'capacity_withheld_mw': self.utilization_analysis.capacity_withheld_mw,
                'constrained_intervals': self.utilization_analysis.maxavail_constrained_intervals,
                'total_intervals': self.utilization_analysis.total_intervals
            }
        }

    def generate_constraint_analysis_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive constraint type analysis report.

        Returns:
        --------
        Dict[str, Any]
            Detailed constraint analysis
        """
        if not self.constraint_analysis:
            return {'error': 'No constraint analysis available'}

        return {
            'constraint_frequency': {
                'counts': self.constraint_analysis.constraint_counts,
                'frequencies_pct': self.constraint_analysis.constraint_frequencies,
                'most_limiting': self.constraint_analysis.most_limiting_constraint
            },
            'constraint_impacts': {
                'revenue_impacts': self.constraint_analysis.constraint_revenue_impacts,
                'efficiency_impacts': self.constraint_analysis.constraint_efficiency_impacts
            },
            'summary': {
                'total_constraint_events': self.constraint_analysis.total_constraint_events,
                'unconstrained_intervals': self.constraint_analysis.unconstrained_intervals,
                'constraint_rate_pct': (self.constraint_analysis.total_constraint_events /
                                        (self.constraint_analysis.total_constraint_events +
                                         self.constraint_analysis.unconstrained_intervals) * 100)
            }
        }

    def generate_strategy_comparison_report(self) -> Dict[str, Any]:
        """Generate a strategy performance comparison report with fixed data access"""
        if not self.strategy_performance:
            self.logger.error("No strategy performance data available")
            return {'error': 'No strategy performance data available'}

        self.logger.debug(f"Generating comparison for strategies: {list(self.strategy_performance.keys())}")

        comparison_data = {}

        for strategy_name, performance in self.strategy_performance.items():
            try:
                comparison_data[strategy_name] = {
                    'revenue_metrics': {
                        'total_revenue': float(performance.total_revenue),
                        'total_dispatched_mwh': float(performance.total_dispatched_mwh),
                        'revenue_per_mwh': float(performance.average_revenue_per_mwh),
                        'capacity_utilization_pct': float(performance.capacity_utilization)
                    },
                    'dispatch_metrics': {
                        'dispatch_frequency_pct': float(performance.dispatch_frequency),
                        'nemde_efficiency_pct': float(performance.nemde_dispatch_efficiency * 100),
                        'successful_intervals': int(performance.successful_dispatch_intervals),
                        'total_intervals': int(performance.total_intervals)
                    },
                    'maxavail_metrics': {
                        'constraint_frequency_pct': float(performance.maxavail_constraint_frequency),
                        'average_utilization_pct': float(performance.average_maxavail_utilization * 100),
                        'revenue_impact': float(performance.maxavail_revenue_impact)
                    },
                    'constraint_analysis': {
                        'constraint_distribution': performance.constraint_type_distribution,
                        'most_common_constraint': performance.most_common_constraint,
                        'constraint_revenue_loss': float(performance.constraint_revenue_loss)
                    }
                }

                self.logger.debug(f"Added comparison data for {strategy_name}")

            except Exception as e:
                self.logger.error(f"Error processing strategy {strategy_name}: {str(e)}")
                continue

        # Calculate rankings with error handling
        rankings = {}
        if len(self.strategy_performance) > 1:
            try:
                strategies = list(self.strategy_performance.keys())

                # Extract values for ranking with error handling
                revenue_values = {}
                efficiency_values = {}
                utilization_values = {}

                for strategy_name in strategies:
                    perf = self.strategy_performance[strategy_name]
                    revenue_values[strategy_name] = float(perf.total_revenue)
                    efficiency_values[strategy_name] = float(perf.nemde_dispatch_efficiency)
                    utilization_values[strategy_name] = float(perf.average_maxavail_utilization)

                rankings = {
                    'by_revenue': sorted(revenue_values.keys(),
                                         key=lambda x: revenue_values[x],
                                         reverse=True),
                    'by_efficiency': sorted(efficiency_values.keys(),
                                            key=lambda x: efficiency_values[x],
                                            reverse=True),
                    'by_utilization': sorted(utilization_values.keys(),
                                             key=lambda x: utilization_values[x],
                                             reverse=True)
                }

                self.logger.debug(f"Rankings calculated: {rankings}")

            except Exception as e:
                self.logger.error(f"Error calculating rankings: {str(e)}")
                rankings = {}

        comparison_data['rankings'] = rankings
        self.logger.info(f"Strategy comparison report generated for {len(comparison_data) - 1} strategies")

        return comparison_data

    def export_dispatch_results(self, output_path: str) -> pd.DataFrame:
        """Export dispatch results to CSV with enhanced error handling"""
        if not self.dispatch_history:
            self.logger.error("No dispatch history to export")
            return pd.DataFrame()

        try:
            # Convert results to DataFrame
            results_data = []
            for result in self.dispatch_history:
                # Format strategy allocation for export
                allocation_str = ','.join(
                    [f"{k}:{v:.2f}" for k, v in
                     result.strategy_allocation.items()]) if result.strategy_allocation else ""

                results_data.append({
                    'timestamp': result.timestamp,
                    'interval_number': result.interval_number,
                    'rrp': result.rrp,
                    'maxavail': getattr(result, 'maxavail_for_interval', 0.0),
                    'total_bandavail': getattr(result, 'total_bandavail', 0.0),
                    'requested_mw': result.requested_mw,
                    'dispatched_mw': result.dispatched_mw,
                    'maxavail_utilization': getattr(result, 'maxavail_utilization', 0.0),
                    'status': result.status.value if hasattr(result.status, 'value') else str(result.status),
                    'constraint_applied': result.constraint_applied,
                    'maxavail_constraint_applied': getattr(result, 'maxavail_constraint_applied', False),
                    'ramp_change': getattr(result, 'ramp_change', 0.0),
                    'revenue': getattr(result, 'revenue', 0.0),
                    'dispatch_efficiency': result.get_dispatch_efficiency(),
                    'dispatched_bands': ','.join(result.dispatched_bands),
                    'remaining_soc': getattr(result, 'remaining_soc', 0.0),
                    'final_override_active': getattr(result, 'final_override_active', False),
                    'strategy_allocation': allocation_str
                })

            results_df = pd.DataFrame(results_data)

            # Try to save with error handling
            try:
                results_df.to_csv(output_path, index=False)
                self.logger.info(f"Dispatch results exported to: {output_path}")
            except PermissionError:
                # Try alternative filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                alt_path = output_path.replace('.csv', f'_{timestamp}.csv')
                try:
                    results_df.to_csv(alt_path, index=False)
                    self.logger.warning(f"Original file locked, saved to: {alt_path}")
                except Exception as e2:
                    self.logger.error(f"Failed to export dispatch results to alternative path: {str(e2)}")
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Failed to export dispatch results: {str(e)}")
                return pd.DataFrame()

            return results_df

        except Exception as e:
            self.logger.error(f"Error creating dispatch results DataFrame: {str(e)}")
            return pd.DataFrame()

    def create_maxavail_utilization_plots(self, output_dir: str) -> List[str]:
        """
        Create comprehensive MAXAVAIL utilization visualization plots.

        Parameters:
        -----------
        output_dir : str
            Directory to save plots

        Returns:
        --------
        List[str]
            List of generated plot file paths
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available - skipping plot generation")
            return []

        if not self.maxavail_events:
            self.logger.warning("No MAXAVAIL events data for plotting")
            return []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        plot_files = []

        # Plot 1: MAXAVAIL Utilization Distribution
        plt.figure(figsize=self.figure_size)
        utilization_data = [event['maxavail_utilization'] for event in self.maxavail_events
                            if event['maxavail_utilization'] is not None]

        if utilization_data:
            plt.hist(utilization_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(utilization_data), color='red', linestyle='--',
                        label=f'Mean: {np.mean(utilization_data):.1%}')
            plt.axvline(np.median(utilization_data), color='green', linestyle='--',
                        label=f'Median: {np.median(utilization_data):.1%}')

            plt.xlabel('MAXAVAIL Utilization Rate')
            plt.ylabel('Frequency')
            plt.title('MAXAVAIL Utilization Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_file = output_path / 'maxavail_utilization_distribution.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))

        # Plot 2: Constraint Frequency by Type
        if self.constraint_analysis:
            plt.figure(figsize=self.figure_size)
            constraints = list(self.constraint_analysis.constraint_counts.keys())
            counts = list(self.constraint_analysis.constraint_counts.values())

            # Exclude "none" for better visualization
            filtered_data = [(c, v) for c, v in zip(constraints, counts) if c != "none"]
            if filtered_data:
                filtered_constraints, filtered_counts = zip(*filtered_data)

                plt.bar(filtered_constraints, filtered_counts, color='lightcoral', alpha=0.7)
                plt.xlabel('Constraint Type')
                plt.ylabel('Number of Events')
                plt.title('Constraint Type Distribution')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)

                plot_file = output_path / 'constraint_type_distribution.png'
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(str(plot_file))

        # Plot 3: MAXAVAIL Utilization Over Time
        if self.maxavail_events:
            plt.figure(figsize=(15, 6))
            timestamps = [event['timestamp'] for event in self.maxavail_events]
            utilizations = [event['maxavail_utilization'] for event in self.maxavail_events]

            plt.plot(timestamps, utilizations, alpha=0.6, color='blue')
            plt.xlabel('Time')
            plt.ylabel('MAXAVAIL Utilization Rate')
            plt.title('MAXAVAIL Utilization Over Time')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            plot_file = output_path / 'maxavail_utilization_timeseries.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))

        self.logger.info(f"Created {len(plot_files)} MAXAVAIL utilization plots in {output_dir}")
        return plot_files

    def export_analysis_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export comprehensive analysis results to files.

        Parameters:
        -----------
        output_dir : str
            Directory to save analysis files

        Returns:
        --------
        Dict[str, str]
            Dictionary of analysis type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Export MAXAVAIL events data
        if self.maxavail_events:
            maxavail_df = pd.DataFrame(self.maxavail_events)
            maxavail_file = output_path / 'maxavail_events_analysis.csv'
            maxavail_df.to_csv(maxavail_file, index=False)
            exported_files['maxavail_events'] = str(maxavail_file)

        # Export MAXAVAIL utilization report
        utilization_report = self.generate_maxavail_utilization_report()
        if 'error' not in utilization_report:
            utilization_file = output_path / 'maxavail_utilization_report.json'
            with open(utilization_file, 'w') as f:
                json.dump(utilization_report, f, indent=2, default=str)
            exported_files['utilization_report'] = str(utilization_file)

        # Export constraint analysis
        constraint_report = self.generate_constraint_analysis_report()
        if 'error' not in constraint_report:
            constraint_file = output_path / 'constraint_analysis_report.json'
            with open(constraint_file, 'w') as f:
                json.dump(constraint_report, f, indent=2, default=str)
            exported_files['constraint_report'] = str(constraint_file)

        # Export strategy comparison
        strategy_report = self.generate_strategy_comparison_report()
        if 'error' not in strategy_report:
            strategy_file = output_path / 'strategy_comparison_report.json'
            with open(strategy_file, 'w') as f:
                json.dump(strategy_report, f, indent=2, default=str)
            exported_files['strategy_report'] = str(strategy_file)

        # Create visualization plots
        plot_files = self.create_maxavail_utilization_plots(str(output_path))
        if plot_files:
            exported_files['plots'] = plot_files

        self.logger.info(f"Exported {len(exported_files)} analysis files to {output_dir}")
        return exported_files

    def get_nemde_dispatch_efficiency_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of NEMDE dispatch efficiency with MAXAVAIL analysis.

        Returns:
        --------
        Dict[str, Any]
            NEMDE efficiency summary with MAXAVAIL metrics
        """
        if not self.dispatch_history:
            return {'error': 'No dispatch history available'}

        # Basic efficiency metrics
        efficiencies = [result.get_dispatch_efficiency() for result in self.dispatch_history]

        # MAXAVAIL-specific efficiency analysis
        maxavail_constrained_efficiencies = []
        unconstrained_efficiencies = []
        maxavail_utilizations = []

        constraint_efficiency_breakdown = {}

        for result in self.dispatch_history:
            efficiency = result.get_dispatch_efficiency()
            constraint = result.constraint_applied or "none"

            # Track efficiency by constraint type
            if constraint not in constraint_efficiency_breakdown:
                constraint_efficiency_breakdown[constraint] = []
            constraint_efficiency_breakdown[constraint].append(efficiency)

            # Separate MAXAVAIL constrained vs unconstrained
            if getattr(result, 'maxavail_constraint_applied', False):
                maxavail_constrained_efficiencies.append(efficiency)
            elif constraint == "none":
                unconstrained_efficiencies.append(efficiency)

            # Collect MAXAVAIL utilization data
            utilization = getattr(result, 'maxavail_utilization', None)
            if utilization is not None:
                maxavail_utilizations.append(utilization)

        # Calculate efficiency statistics by constraint type
        constraint_efficiency_stats = {}
        for constraint, eff_list in constraint_efficiency_breakdown.items():
            if eff_list:
                constraint_efficiency_stats[constraint] = {
                    'count': len(eff_list),
                    'average_pct': np.mean(eff_list) * 100,
                    'median_pct': np.median(eff_list) * 100,
                    'min_pct': np.min(eff_list) * 100,
                    'max_pct': np.max(eff_list) * 100
                }

        # MAXAVAIL utilization correlation with efficiency
        efficiency_utilization_correlation = 0.0
        if len(efficiencies) == len(maxavail_utilizations) and len(maxavail_utilizations) > 1:
            efficiency_utilization_correlation = float(np.corrcoef(efficiencies, maxavail_utilizations)[0, 1])

        return {
            'overall_efficiency': {
                'average_efficiency_pct': np.mean(efficiencies) * 100,
                'median_efficiency_pct': np.median(efficiencies) * 100,
                'efficiency_std': np.std(efficiencies),
                'min_efficiency_pct': np.min(efficiencies) * 100,
                'max_efficiency_pct': np.max(efficiencies) * 100,
                'efficiency_percentiles': {
                    f"p{p}": np.percentile(efficiencies, p) * 100
                    for p in [10, 25, 50, 75, 90, 95]
                }
            },
            'maxavail_efficiency_analysis': {
                'maxavail_constrained_count': len(maxavail_constrained_efficiencies),
                'unconstrained_count': len(unconstrained_efficiencies),
                'maxavail_constrained_avg_pct': np.mean(
                    maxavail_constrained_efficiencies) * 100 if maxavail_constrained_efficiencies else 0,
                'unconstrained_avg_pct': np.mean(unconstrained_efficiencies) * 100 if unconstrained_efficiencies else 0,
                'efficiency_gap_pct': (np.mean(unconstrained_efficiencies) - np.mean(
                    maxavail_constrained_efficiencies)) * 100 if maxavail_constrained_efficiencies and unconstrained_efficiencies else 0
            },
            'constraint_efficiency_breakdown': constraint_efficiency_stats,
            'maxavail_utilization_metrics': {
                'average_utilization_pct': np.mean(maxavail_utilizations) * 100 if maxavail_utilizations else 0,
                'utilization_efficiency_correlation': efficiency_utilization_correlation,
                'total_intervals_analyzed': len(maxavail_utilizations)
            },
            'efficiency_quality_indicators': {
                'high_efficiency_intervals_pct': sum(1 for e in efficiencies if e > 0.95) / len(efficiencies) * 100,
                'low_efficiency_intervals_pct': sum(1 for e in efficiencies if e < 0.5) / len(efficiencies) * 100,
                'perfect_dispatch_intervals': sum(1 for e in efficiencies if e >= 0.999),
                'failed_dispatch_intervals': sum(1 for e in efficiencies if e == 0.0)
            }
        }

    def analyze_maxavail_constraint_patterns(self) -> Dict[str, Any]:
        """
        Analyze temporal and operational patterns in MAXAVAIL constraints.

        Returns:
        --------
        Dict[str, Any]
            MAXAVAIL constraint pattern analysis
        """
        if not self.maxavail_events:
            return {'error': 'No MAXAVAIL events data available'}

        # Extract constrained events only
        constrained_events = [event for event in self.maxavail_events
                              if event['maxavail_constraint_applied']]

        if not constrained_events:
            return {'message': 'No MAXAVAIL constraint events found'}

        # Temporal pattern analysis
        hourly_constraints = {}
        interval_constraints = {}
        daily_constraints = {}

        for event in constrained_events:
            timestamp = event['timestamp']
            hour = timestamp.hour
            interval = event['interval_number']
            date = timestamp.date()

            hourly_constraints[hour] = hourly_constraints.get(hour, 0) + 1
            interval_constraints[interval] = interval_constraints.get(interval, 0) + 1
            daily_constraints[date] = daily_constraints.get(date, 0) + 1

        # RRP correlation analysis
        rrp_values = [float(event['rrp']) for event in constrained_events]
        utilization_values = [float(event['maxavail_utilization']) for event in constrained_events]

        rrp_utilization_correlation = 0.0
        if len(rrp_values) > 1 and len(utilization_values) > 1:
            rrp_utilization_correlation = float(np.corrcoef(rrp_values, utilization_values)[0, 1])

        # Constraint severity analysis
        severe_constraints = sum(1 for event in constrained_events if event['maxavail_utilization'] < 0.5)
        moderate_constraints = sum(1 for event in constrained_events if 0.5 <= event['maxavail_utilization'] < 0.8)
        mild_constraints = sum(1 for event in constrained_events if event['maxavail_utilization'] >= 0.8)

        return {
            'constraint_frequency': {
                'total_constrained_events': len(constrained_events),
                'constraint_rate_pct': len(constrained_events) / len(self.maxavail_events) * 100,
                'unique_constrained_hours': len(hourly_constraints),
                'unique_constrained_intervals': len(interval_constraints),
                'unique_constrained_days': len(daily_constraints)
            },
            'temporal_patterns': {
                'hourly_distribution': hourly_constraints,
                'interval_distribution': interval_constraints,
                'peak_constraint_hour': max(hourly_constraints.keys(),
                                            key=lambda x: hourly_constraints[x]) if hourly_constraints else None,
                'peak_constraint_interval': max(interval_constraints.keys(),
                                                key=lambda x: interval_constraints[x]) if interval_constraints else None
            },
            'severity_analysis': {
                'severe_constraints': severe_constraints,
                'moderate_constraints': moderate_constraints,
                'mild_constraints': mild_constraints,
                'average_utilization_when_constrained_pct': np.mean(
                    utilization_values) * 100 if utilization_values else 0
            },
            'economic_correlation': {
                'average_rrp_when_constrained': float(np.mean(rrp_values)) if rrp_values else 0.0,
                'rrp_utilization_correlation': rrp_utilization_correlation,
                'high_price_constraints': len([event for event in constrained_events if event['rrp'] > 100]),
                'low_price_constraints': len([event for event in constrained_events if event['rrp'] < 50])
            }
        }

    def compare_strategy_maxavail_performance(self) -> Dict[str, Any]:
        """
        Compare MAXAVAIL performance across different strategies.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific MAXAVAIL performance comparison
        """
        if not self.strategy_performance:
            return {'error': 'No strategy performance data available'}

        if len(self.strategy_performance) < 2:
            return {'message': 'Need multiple strategies for comparison'}

        comparison_metrics = {}

        for strategy_name, performance in self.strategy_performance.items():
            # Extract strategy-specific dispatch results
            strategy_results = [result for result in self.dispatch_history
                                if getattr(result, 'strategy_name', 'default_strategy') == strategy_name]

            if not strategy_results:
                continue

            # MAXAVAIL-specific metrics for this strategy
            maxavail_events = sum(1 for r in strategy_results
                                  if getattr(r, 'maxavail_constraint_applied', False))

            maxavail_utilizations = [getattr(r, 'maxavail_utilization', 0.0)
                                     for r in strategy_results
                                     if getattr(r, 'maxavail_utilization', None) is not None]

            avg_utilization = np.mean(maxavail_utilizations) if maxavail_utilizations else 0.0

            # Efficiency metrics
            efficiencies = [r.get_dispatch_efficiency() for r in strategy_results]
            avg_efficiency = np.mean(efficiencies) if efficiencies else 0.0

            # Revenue per MW capacity (normalized comparison)
            revenue_per_mw = performance.total_revenue / 100  # Assuming 100MW capacity

            comparison_metrics[strategy_name] = {
                'maxavail_constraint_frequency_pct': performance.maxavail_constraint_frequency,
                'average_maxavail_utilization_pct': avg_utilization * 100,
                'average_dispatch_efficiency_pct': avg_efficiency * 100,
                'revenue_per_mw': revenue_per_mw,
                'total_intervals': len(strategy_results),
                'successful_dispatches': sum(1 for r in strategy_results if r.dispatched_mw > 0),
                'constraint_events': maxavail_events,
                'revenue_per_mwh': performance.average_revenue_per_mwh
            }

        # Calculate rankings
        strategies = list(comparison_metrics.keys())

        rankings = {
            'by_maxavail_utilization': sorted(strategies,
                                              key=lambda x: comparison_metrics[x]['average_maxavail_utilization_pct'],
                                              reverse=True),
            'by_dispatch_efficiency': sorted(strategies,
                                             key=lambda x: comparison_metrics[x]['average_dispatch_efficiency_pct'],
                                             reverse=True),
            'by_revenue_per_mw': sorted(strategies,
                                        key=lambda x: comparison_metrics[x]['revenue_per_mw'],
                                        reverse=True),
            'by_constraint_frequency': sorted(strategies,
                                              key=lambda x: comparison_metrics[x]['maxavail_constraint_frequency_pct'])
        }

        # Identify best performing strategy overall
        best_strategy_scores = {}
        for strategy in strategies:
            metrics = comparison_metrics[strategy]
            # Composite score (higher utilization, higher efficiency, lower constraint frequency)
            score = (metrics['average_maxavail_utilization_pct'] +
                     metrics['average_dispatch_efficiency_pct'] -
                     metrics['maxavail_constraint_frequency_pct'])
            best_strategy_scores[strategy] = score

        best_overall_strategy = max(best_strategy_scores.keys(),
                                    key=lambda x: best_strategy_scores[x]) if best_strategy_scores else "unknown"

        return {
            'strategy_metrics': comparison_metrics,
            'rankings': rankings,
            'best_overall_strategy': best_overall_strategy,
            'performance_summary': {
                'highest_utilization': rankings['by_maxavail_utilization'][0],
                'highest_efficiency': rankings['by_dispatch_efficiency'][0],
                'highest_revenue': rankings['by_revenue_per_mw'][0],
                'lowest_constraints': rankings['by_constraint_frequency'][0]
            }
        }

    def generate_maxavail_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Generate optimization recommendations based on MAXAVAIL analysis.

        Returns:
        --------
        Dict[str, Any]
            MAXAVAIL optimization recommendations
        """
        if not self.utilization_analysis or not self.constraint_analysis:
            return {'error': 'Insufficient analysis data for recommendations'}

        recommendations = {
            'capacity_optimization': [],
            'bidding_strategy': [],
            'operational_improvements': [],
            'risk_management': []
        }

        # Capacity optimization recommendations
        if self.utilization_analysis.average_utilization < 0.7:
            recommendations['capacity_optimization'].append({
                'issue': 'Low MAXAVAIL utilization',
                'current_value': f"{self.utilization_analysis.average_utilization:.1%}",
                'recommendation': 'Consider more aggressive bidding strategies to increase capacity utilization',
                'priority': 'High'
            })

        if self.utilization_analysis.capacity_withheld_mw > 50:  # Assuming 100MW battery
            recommendations['capacity_optimization'].append({
                'issue': 'High capacity withholding',
                'current_value': f"{self.utilization_analysis.capacity_withheld_mw:.1f} MW",
                'recommendation': 'Review price band allocation to reduce capacity withholding',
                'priority': 'Medium'
            })

        # Bidding strategy recommendations
        if self.utilization_analysis.constraint_frequency_pct > 20:
            recommendations['bidding_strategy'].append({
                'issue': 'High MAXAVAIL constraint frequency',
                'current_value': f"{self.utilization_analysis.constraint_frequency_pct:.1f}%",
                'recommendation': 'Implement dynamic MAXAVAIL-aware bidding to reduce constraint impacts',
                'priority': 'High'
            })

        if self.constraint_analysis.most_limiting_constraint == 'maxavail':
            recommendations['bidding_strategy'].append({
                'issue': 'MAXAVAIL is primary constraint',
                'current_value': f"Most limiting constraint: {self.constraint_analysis.most_limiting_constraint}",
                'recommendation': 'Focus on MAXAVAIL optimization over other constraint types',
                'priority': 'High'
            })

        # Operational improvements
        if self.utilization_analysis.average_nemde_efficiency < 0.8:
            recommendations['operational_improvements'].append({
                'issue': 'Low NEMDE dispatch efficiency',
                'current_value': f"{self.utilization_analysis.average_nemde_efficiency:.1%}",
                'recommendation': 'Improve bid schedule alignment with MAXAVAIL constraints',
                'priority': 'Medium'
            })

        if self.utilization_analysis.dispatch_success_rate < 70:
            recommendations['operational_improvements'].append({
                'issue': 'Low dispatch success rate',
                'current_value': f"{self.utilization_analysis.dispatch_success_rate:.1f}%",
                'recommendation': 'Review bidding strategy to increase successful dispatch frequency',
                'priority': 'Medium'
            })

        # Risk management recommendations
        if self.utilization_analysis.revenue_impact_pct > 15:
            recommendations['risk_management'].append({
                'issue': 'High revenue impact from constraints',
                'current_value': f"{self.utilization_analysis.revenue_impact_pct:.1f}%",
                'recommendation': 'Implement constraint-aware revenue protection strategies',
                'priority': 'High'
            })

        # Calculate overall optimization score
        optimization_score = self._calculate_optimization_score()

        return {
            'recommendations': recommendations,
            'optimization_score': optimization_score,
            'summary': {
                'total_recommendations': sum(len(recs) for recs in recommendations.values()),
                'high_priority_items': sum(1 for rec_list in recommendations.values()
                                           for rec in rec_list if rec['priority'] == 'High'),
                'key_focus_areas': [area for area, recs in recommendations.items() if recs]
            }
        }

    def _calculate_optimization_score(self) -> Dict[str, float]:
        """Calculate overall optimization score based on MAXAVAIL metrics"""
        if not self.utilization_analysis:
            return {'overall_score': 0.0}

        # Weight different metrics for optimization score
        utilization_score = self.utilization_analysis.average_utilization * 30
        efficiency_score = self.utilization_analysis.average_nemde_efficiency * 25
        constraint_score = max(0, (1 - self.utilization_analysis.constraint_frequency_pct / 100)) * 25
        success_score = self.utilization_analysis.dispatch_success_rate / 100 * 20

        overall_score = utilization_score + efficiency_score + constraint_score + success_score

        return {
            'overall_score': overall_score,
            'utilization_component': utilization_score,
            'efficiency_component': efficiency_score,
            'constraint_component': constraint_score,
            'success_component': success_score,
            'score_interpretation': self._interpret_optimization_score(overall_score)
        }

    def _interpret_optimization_score(self, score: float) -> str:
        """Interpret optimization score"""
        if score >= 80:
            return "Excellent - MAXAVAIL optimization is highly effective"
        elif score >= 65:
            return "Good - MAXAVAIL optimization is working well with room for improvement"
        elif score >= 50:
            return "Average - MAXAVAIL optimization needs attention"
        elif score >= 35:
            return "Poor - Significant MAXAVAIL optimization issues identified"
        else:
            return "Critical - Major MAXAVAIL optimization problems require immediate attention"

    def create_comprehensive_report(self):
        """Create a comprehensive analysis report with fixed strategy identification"""

        @dataclass
        class StrategyComparison:
            best_revenue_strategy: str
            best_efficiency_strategy: str
            performance_matrix: pd.DataFrame
            risk_return_analysis: Dict[str, Any]

        @dataclass
        class MarketAnalysis:
            price_volatility_analysis: Dict[str, Any]
            peak_period_analysis: Dict[str, Any]

        @dataclass
        class ComprehensiveAnalysisResults:
            strategy_comparison: StrategyComparison
            market_analysis: MarketAnalysis
            recommendations: List[str]
            maxavail_analysis: Dict[str, Any]

        self.logger.debug("Creating comprehensive report...")

        # Generate strategy comparison with better error handling
        strategy_comparison_data = self.generate_strategy_comparison_report()

        if 'error' in strategy_comparison_data:
            self.logger.error("Strategy comparison failed, creating empty results")
            strategy_comparison = StrategyComparison(
                best_revenue_strategy="unknown",
                best_efficiency_strategy="unknown",
                performance_matrix=pd.DataFrame(),
                risk_return_analysis={}
            )
        else:
            rankings = strategy_comparison_data.get('rankings', {})

            # Extract best strategies with error handling
            best_revenue = "unknown"
            best_efficiency = "unknown"

            if rankings.get('by_revenue'):
                best_revenue = rankings['by_revenue'][0]
                self.logger.debug(f"Best revenue strategy: {best_revenue}")

            if rankings.get('by_efficiency'):
                best_efficiency = rankings['by_efficiency'][0]
                self.logger.debug(f"Best efficiency strategy: {best_efficiency}")

            # Create performance matrix from strategy_performance directly
            performance_matrix = pd.DataFrame()
            try:
                if self.strategy_performance:
                    matrix_data = {}
                    for strategy_name, perf in self.strategy_performance.items():
                        matrix_data[strategy_name] = {
                            'Total Revenue': float(perf.total_revenue),
                            'Revenue per MWh': float(perf.average_revenue_per_mwh),
                            'Dispatch Efficiency': float(perf.nemde_dispatch_efficiency),
                            'MAXAVAIL Utilization': float(perf.average_maxavail_utilization),
                            'Constraint Frequency': float(perf.maxavail_constraint_frequency)
                        }

                    performance_matrix = pd.DataFrame(matrix_data).T
                    self.logger.debug(f"Performance matrix created with shape: {performance_matrix.shape}")

            except Exception as e:
                self.logger.error(f"Error creating performance matrix: {str(e)}")
                performance_matrix = pd.DataFrame()

            strategy_comparison = StrategyComparison(
                best_revenue_strategy=best_revenue,
                best_efficiency_strategy=best_efficiency,
                performance_matrix=performance_matrix,
                risk_return_analysis={}
            )

        # Generate market analysis
        market_analysis = MarketAnalysis(
            price_volatility_analysis={'daily_volatility': 0.0},
            peak_period_analysis={}
        )

        # Generate recommendations
        optimization_recs = self.generate_maxavail_optimization_recommendations()
        recommendations = []

        if 'error' not in optimization_recs:
            for category, recs in optimization_recs.get('recommendations', {}).items():
                for rec in recs:
                    recommendations.append(f"{category}: {rec['recommendation']}")

        # Get MAXAVAIL analysis
        maxavail_analysis = self.generate_maxavail_utilization_report()

        self.logger.info("Comprehensive report created successfully")

        return ComprehensiveAnalysisResults(
            strategy_comparison=strategy_comparison,
            market_analysis=market_analysis,
            recommendations=recommendations,
            maxavail_analysis=maxavail_analysis
        )

    def export_detailed_report(self, file_path: str, comprehensive_results) -> None:
        """
        Export detailed report to text file as expected by main.py.
        """
        try:
            with open(file_path, 'w') as f:
                f.write("BATTERY BIDDING BACKTEST - DETAILED ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")

                f.write("STRATEGY COMPARISON\n")
                f.write("-" * 30 + "\n")
                f.write(f"Best Revenue Strategy: {comprehensive_results.strategy_comparison.best_revenue_strategy}\n")
                f.write(
                    f"Best Efficiency Strategy: {comprehensive_results.strategy_comparison.best_efficiency_strategy}\n\n")

                if not comprehensive_results.strategy_comparison.performance_matrix.empty:
                    f.write("Performance Matrix:\n")
                    f.write(comprehensive_results.strategy_comparison.performance_matrix.to_string())
                    f.write("\n\n")

                f.write("MAXAVAIL ANALYSIS\n")
                f.write("-" * 30 + "\n")
                maxavail_data = comprehensive_results.maxavail_analysis
                if 'error' not in maxavail_data:
                    summary = maxavail_data.get('summary', {})
                    for key, value in summary.items():
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")

                f.write("RECOMMENDATIONS\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(comprehensive_results.recommendations, 1):
                    f.write(f"{i}. {rec}\n")

            self.logger.info(f"Detailed report exported to: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to export detailed report: {str(e)}")

    def export_results_summary(self, file_path: str, comprehensive_results) -> Dict[str, Any]:
        """
        Export results summary to JSON as expected by main.py.
        """
        try:
            summary_data = {
                'best_strategies': {
                    'revenue': comprehensive_results.strategy_comparison.best_revenue_strategy,
                    'efficiency': comprehensive_results.strategy_comparison.best_efficiency_strategy
                },
                'recommendations_count': len(comprehensive_results.recommendations),
                'maxavail_analysis': comprehensive_results.maxavail_analysis,
                'analysis_timestamp': datetime.now().isoformat()
            }

            with open(file_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)

            self.logger.info(f"Results summary exported to: {file_path}")
            return summary_data

        except Exception as e:
            self.logger.error(f"Failed to export results summary: {str(e)}")
            return {}

    def generate_visualizations(self, output_dir: str, comprehensive_results) -> Dict[str, str]:
        """
        Generate visualizations as expected by main.py.
        """
        try:
            plot_files = self.create_maxavail_utilization_plots(output_dir)

            visualization_files = {}
            for i, file_path in enumerate(plot_files):
                file_name = Path(file_path).stem
                visualization_files[file_name] = file_path

            return visualization_files

        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {str(e)}")
            return {}

    def _ensure_strategy_attribution(self) -> None:
        """Ensure all dispatch results have proper strategy attribution"""
        # Extract strategy names from backtest_results if available
        strategy_names = []
        if 'daily_results' in self.backtest_results:
            for daily_data in self.backtest_results['daily_results'].values():
                if 'strategy_results' in daily_data:
                    strategy_names.extend(daily_data['strategy_results'].keys())

        # Remove duplicates
        strategy_names = list(set(strategy_names))
        self.logger.debug(f"Available strategies from backtest results: {strategy_names}")

        # If we have strategy names and dispatch history, try to attribute properly
        if strategy_names and self.dispatch_history:
            # Simple attribution: assume results are in order of strategies
            results_per_strategy = len(self.dispatch_history) // len(strategy_names)

            for i, result in enumerate(self.dispatch_history):
                if not hasattr(result, 'strategy_name') or not result.strategy_name:
                    strategy_index = min(i // results_per_strategy, len(strategy_names) - 1)
                    result.strategy_name = strategy_names[strategy_index]

            self.logger.debug("Strategy attribution completed")

    def compare_with_benchmark(self, benchmark_strategy: str) -> Dict[str, Any]:
        """
        Compare strategies with benchmark as expected by main.py.
        """
        try:
            if not self.strategy_performance:
                return {'error': 'No strategy performance data available'}

            if benchmark_strategy not in self.strategy_performance:
                return {'error': f'Benchmark strategy {benchmark_strategy} not found'}

            benchmark_perf = self.strategy_performance[benchmark_strategy]

            comparison = {
                'benchmark_strategy': benchmark_strategy,
                'benchmark_performance': benchmark_perf,
                'comparisons': {}
            }

            for strategy_name, performance in self.strategy_performance.items():
                if strategy_name != benchmark_strategy:
                    comparison['comparisons'][strategy_name] = {
                        'revenue_difference': performance.total_revenue - benchmark_perf.total_revenue,
                        'efficiency_difference': performance.nemde_dispatch_efficiency - benchmark_perf.nemde_dispatch_efficiency,
                        'utilization_difference': performance.average_maxavail_utilization - benchmark_perf.average_maxavail_utilization
                    }

            return comparison

        except Exception as e:
            self.logger.error(f"Benchmark comparison failed: {str(e)}")
            return {'error': str(e)}

    def reset_analyzer(self) -> None:
        """Reset analyzer state for new analysis"""
        self.backtest_results.clear()
        self.dispatch_history.clear()
        self.revenue_breakdown_history.clear()
        self.strategy_performance.clear()
        self.maxavail_events.clear()
        self.constraint_analysis = None
        self.utilization_analysis = None

        self.logger.info("ResultsAnalyzer reset completed")
