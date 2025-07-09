import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ConstraintType(Enum):
    """Types of dispatch constraints that can be applied"""
    NONE = "none"
    MAXAVAIL = "maxavail"
    PRICE_THRESHOLD = "price_threshold"
    RAMP_LIMIT = "ramp_limit"
    RAMP_LIMIT_POST_NEMDE = "ramp_limit_post_nemde"
    POWER_LIMIT = "power_limit"
    SOC_FINAL_CHECK = "soc_final_check"
    NEGATIVE_DISPATCH = "negative_dispatch"
    OTHER = "other"


@dataclass
class IntervalRevenue:
    """Revenue details for a single dispatch interval"""
    interval_datetime: datetime
    interval_number: int
    dispatched_mwh: float
    rrp: float
    gross_revenue: float
    net_revenue: float
    strategy_name: str

    # MAXAVAIL-specific metrics
    maxavail: float = 0.0
    requested_capacity: float = 0.0
    constraint_applied: str = "none"
    dispatch_efficiency: float = 1.0
    revenue_efficiency: float = 1.0
    estimated_revenue_loss: float = 0.0


@dataclass
class MAXAVAILMetrics:
    """Comprehensive MAXAVAIL constraint metrics"""
    total_intervals: int = 0
    constrained_intervals: int = 0
    constraint_frequency: float = 0.0  # % of intervals constrained

    # Capacity metrics
    total_requested_mw: float = 0.0
    total_maxavail_mw: float = 0.0
    total_dispatched_mw: float = 0.0

    # Efficiency metrics
    average_maxavail_utilization: float = 0.0
    average_nemde_efficiency: float = 0.0
    capacity_withheld_mw: float = 0.0  # MW lost to MAXAVAIL constraints

    # Revenue impact
    estimated_revenue_loss: float = 0.0  # Revenue lost due to MAXAVAIL constraints
    constraint_revenue_impact: float = 0.0  # % revenue impact

    # Constraint type distribution
    constraint_type_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class RevenueBreakdown:
    """Enhanced revenue breakdown with MAXAVAIL impact analysis"""
    # Core revenue components
    energy_revenue: float
    capacity_revenue: float
    ancillary_revenue: float
    total_gross_revenue: float
    transmission_losses: float
    total_net_revenue: float

    # Performance metrics
    revenue_per_mwh: float
    revenue_per_mw_capacity: float
    capacity_factor: float
    utilization_rate: float

    # MAXAVAIL-specific metrics
    maxavail_metrics: MAXAVAILMetrics = field(default_factory=MAXAVAILMetrics)
    dispatch_efficiency: float = 0.0  # Overall dispatch vs requested efficiency
    constraint_adjusted_revenue: float = 0.0  # Revenue adjusted for constraints


class RevenueCalculator:
    """
    Enhanced revenue calculator with comprehensive MAXAVAIL constraint analysis.

    Calculates revenue while tracking the impact of MAXAVAIL constraints on
    dispatch efficiency and revenue realization. Integrates with existing
    DispatchResult structure from DispatchSimulator.
    """

    def __init__(self, config_or_capacity,
                 transmission_loss_factor: float = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize revenue calculator with flexible constructor.

        Parameters:
        -----------
        config_or_capacity : Configuration or float
            Either a Configuration object or max capacity in MW
        transmission_loss_factor : float, optional
            TLF value (only used if first param is float)
        logger : logging.Logger, optional
            Logger instance
        """
        # Handle both constructor patterns
        if hasattr(config_or_capacity, 'battery_parameters'):
            # Configuration object passed (as expected by main.py)
            config = config_or_capacity
            self.max_capacity_mw = config.battery_parameters['max_discharge_power']
            self.transmission_loss_factor = config.battery_parameters.get('TLF', 1.0)
        else:
            # Legacy pattern: float capacity passed
            self.max_capacity_mw = config_or_capacity
            self.transmission_loss_factor = transmission_loss_factor or 1.0

        self.logger = logger or logging.getLogger(__name__)

        # Revenue tracking
        self.daily_revenues = {}
        self.interval_revenues = []

        # MAXAVAIL analysis tracking
        self.maxavail_events = []
        self.constraint_impact_history = []

    def calculate_interval_revenue(self, dispatch_result) -> IntervalRevenue:
        """
        Calculate revenue for a single dispatch interval with MAXAVAIL impact analysis.

        Parameters:
        -----------
        dispatch_result : DispatchResult
            Dispatch result from DispatchSimulator with MAXAVAIL data

        Returns:
        --------
        IntervalRevenue
            Detailed interval revenue with constraint impact
        """
        # Convert MW to MWh for 5-minute intervals
        dispatched_mwh = dispatch_result.dispatched_mw / 12  # 5 minutes = 1/12 hour

        # Basic revenue calculation (already includes TLF in DispatchSimulator)
        gross_revenue = dispatched_mwh * dispatch_result.rrp
        net_revenue = dispatch_result.revenue if dispatch_result.revenue is not None else gross_revenue

        # MAXAVAIL impact analysis
        constraint_impact = self._analyze_constraint_impact(dispatch_result)

        # Create interval revenue object
        interval_revenue = IntervalRevenue(
            interval_datetime=dispatch_result.timestamp,
            interval_number=dispatch_result.interval_number,
            dispatched_mwh=dispatched_mwh,
            rrp=dispatch_result.rrp,
            gross_revenue=gross_revenue,
            net_revenue=net_revenue,
            strategy_name="strategy",  # Will be updated by caller if needed

            # MAXAVAIL-specific metrics
            maxavail=dispatch_result.maxavail_for_interval or 0.0,
            requested_capacity=dispatch_result.requested_mw,
            constraint_applied=dispatch_result.constraint_applied or "none",
            dispatch_efficiency=dispatch_result.get_dispatch_efficiency(),
            revenue_efficiency=constraint_impact['revenue_efficiency'],
            estimated_revenue_loss=constraint_impact['estimated_revenue_loss']
        )

        # Store for analysis
        self.interval_revenues.append(interval_revenue)

        return interval_revenue

    def _analyze_constraint_impact(self, dispatch_result) -> Dict[str, float]:
        """
        Analyze the revenue impact of MAXAVAIL and other constraints.

        Parameters:
        -----------
        dispatch_result : DispatchResult
            Dispatch result with constraint information

        Returns:
        --------
        Dict[str, float]
            Constraint impact metrics
        """
        if not dispatch_result.constraint_applied or dispatch_result.constraint_applied == "none":
            return {
                'revenue_efficiency': 1.0,
                'estimated_revenue_loss': 0.0,
                'capacity_utilization_loss': 0.0
            }

        # Calculate potential revenue if unconstrained
        if dispatch_result.requested_mw > 0:
            unconstrained_mwh = dispatch_result.requested_mw / 12  # Convert MW to MWh
            unconstrained_revenue = (unconstrained_mwh *
                                     dispatch_result.rrp *
                                     self.transmission_loss_factor)

            actual_mwh = dispatch_result.dispatched_mw / 12
            actual_revenue = (actual_mwh *
                              dispatch_result.rrp *
                              self.transmission_loss_factor)

            estimated_revenue_loss = unconstrained_revenue - actual_revenue
            revenue_efficiency = actual_revenue / unconstrained_revenue if unconstrained_revenue > 0 else 0
        else:
            estimated_revenue_loss = 0.0
            revenue_efficiency = 1.0

        capacity_utilization_loss = dispatch_result.requested_mw - dispatch_result.dispatched_mw

        # Track constraint event with enhanced classification
        constraint_type = self._classify_constraint_type(dispatch_result.constraint_applied)

        constraint_event = {
            'interval_datetime': dispatch_result.timestamp,
            'interval_number': dispatch_result.interval_number,
            'constraint_type': constraint_type,
            'constraint_applied': dispatch_result.constraint_applied,
            'requested_mw': dispatch_result.requested_mw,
            'maxavail_mw': dispatch_result.maxavail_for_interval or 0.0,
            'dispatched_mw': dispatch_result.dispatched_mw,
            'capacity_loss_mw': capacity_utilization_loss,
            'revenue_loss': estimated_revenue_loss,
            'rrp': dispatch_result.rrp,
            'maxavail_utilization': dispatch_result.maxavail_utilization or 0.0,
            'total_bandavail': dispatch_result.total_bandavail or 0.0,
            'maxavail_constraint_applied': dispatch_result.maxavail_constraint_applied or False
        }
        self.maxavail_events.append(constraint_event)

        return {
            'revenue_efficiency': revenue_efficiency,
            'estimated_revenue_loss': estimated_revenue_loss,
            'capacity_utilization_loss': capacity_utilization_loss
        }

    def _classify_constraint_type(self, constraint_applied: str) -> ConstraintType:
        """
        Classify constraint type from DispatchSimulator constraint strings.

        Parameters:
        -----------
        constraint_applied : str
            Constraint type string from DispatchSimulator

        Returns:
        --------
        ConstraintType
            Classified constraint type
        """
        if not constraint_applied or constraint_applied == "none":
            return ConstraintType.NONE

        constraint_mapping = {
            'maxavail': ConstraintType.MAXAVAIL,
            'price_threshold': ConstraintType.PRICE_THRESHOLD,
            'ramp_limit': ConstraintType.RAMP_LIMIT,
            'ramp_limit_post_nemde': ConstraintType.RAMP_LIMIT_POST_NEMDE,
            'power_limit': ConstraintType.POWER_LIMIT,
            'soc_final_check': ConstraintType.SOC_FINAL_CHECK,
            'negative_dispatch': ConstraintType.NEGATIVE_DISPATCH
        }

        return constraint_mapping.get(constraint_applied, ConstraintType.OTHER)

    def calculate_daily_revenue(self, dispatch_results: List,
                                date: datetime, strategy_name: str = "strategy") -> RevenueBreakdown:
        """
        Calculate comprehensive daily revenue breakdown with MAXAVAIL impact analysis.

        Parameters:
        -----------
        dispatch_results : List[DispatchResult]
            List of dispatch results from DispatchSimulator for the day
        date : datetime
            Date for the calculation
        strategy_name : str
            Name of the strategy being analyzed

        Returns:
        --------
        RevenueBreakdown
            Detailed revenue breakdown with MAXAVAIL metrics
        """
        if not dispatch_results:
            return self._create_zero_revenue_breakdown(date)

        # Calculate interval revenues with MAXAVAIL analysis
        daily_interval_revenues = []
        for result in dispatch_results:
            interval_rev = self.calculate_interval_revenue(result)
            interval_rev.strategy_name = strategy_name  # Update strategy name
            daily_interval_revenues.append(interval_rev)

        # Aggregate daily totals
        total_dispatched_mwh = sum(ir.dispatched_mwh for ir in daily_interval_revenues)
        total_gross_revenue = sum(ir.gross_revenue for ir in daily_interval_revenues)
        total_net_revenue = sum(ir.net_revenue for ir in daily_interval_revenues)
        transmission_losses = total_gross_revenue - total_net_revenue

        # Calculate performance metrics
        revenue_per_mwh = total_net_revenue / total_dispatched_mwh if total_dispatched_mwh > 0 else 0
        revenue_per_mw_capacity = total_net_revenue / self.max_capacity_mw if self.max_capacity_mw > 0 else 0

        # Calculate capacity factor (for daily peak period)
        peak_period_hours = 2.5  # 17:00-19:30
        theoretical_max_mwh = self.max_capacity_mw * peak_period_hours
        capacity_factor = (total_dispatched_mwh / theoretical_max_mwh * 100) if theoretical_max_mwh > 0 else 0

        # Calculate utilization rate (% of available capacity used)
        total_dispatched_mw = sum(result.dispatched_mw for result in dispatch_results)
        max_possible_dispatch = self.max_capacity_mw * len(dispatch_results)
        utilization_rate = (total_dispatched_mw / max_possible_dispatch * 100) if max_possible_dispatch > 0 else 0

        # Calculate MAXAVAIL-specific metrics
        maxavail_metrics = self._calculate_maxavail_metrics(dispatch_results, daily_interval_revenues)

        # Calculate overall dispatch efficiency
        total_requested = sum(result.requested_mw for result in dispatch_results)
        dispatch_efficiency = (total_dispatched_mw / total_requested * 100) if total_requested > 0 else 100

        # Calculate constraint-adjusted revenue (what revenue would have been without constraints)
        total_revenue_loss = sum(ir.estimated_revenue_loss for ir in daily_interval_revenues)
        constraint_adjusted_revenue = total_net_revenue + total_revenue_loss

        # Create enhanced revenue breakdown
        revenue_breakdown = RevenueBreakdown(
            energy_revenue=total_net_revenue,
            capacity_revenue=0.0,  # Future enhancement
            ancillary_revenue=0.0,  # Future enhancement
            total_gross_revenue=total_gross_revenue,
            transmission_losses=transmission_losses,
            total_net_revenue=total_net_revenue,
            revenue_per_mwh=revenue_per_mwh,
            revenue_per_mw_capacity=revenue_per_mw_capacity,
            capacity_factor=capacity_factor,
            utilization_rate=utilization_rate,

            # MAXAVAIL-enhanced metrics
            maxavail_metrics=maxavail_metrics,
            dispatch_efficiency=dispatch_efficiency,
            constraint_adjusted_revenue=constraint_adjusted_revenue
        )

        # Store daily revenue
        date_key = date.strftime('%Y-%m-%d')
        self.daily_revenues[date_key] = total_net_revenue

        # Enhanced logging with MAXAVAIL metrics
        self.logger.info(f"Daily revenue calculated for {date_key}: ${total_net_revenue:.2f} "
                         f"({total_dispatched_mwh:.1f} MWh, CF: {capacity_factor:.1f}%, "
                         f"Dispatch Eff: {dispatch_efficiency:.1f}%, "
                         f"MAXAVAIL Constrained: {maxavail_metrics.constraint_frequency:.1f}%)")

        if maxavail_metrics.estimated_revenue_loss > 0:
            self.logger.warning(f"MAXAVAIL revenue impact for {date_key}: "
                                f"-${maxavail_metrics.estimated_revenue_loss:.2f} "
                                f"({maxavail_metrics.constraint_revenue_impact:.1f}%)")

        return revenue_breakdown

    def _calculate_maxavail_metrics(self, dispatch_results: List,
                                    interval_revenues: List[IntervalRevenue]) -> MAXAVAILMetrics:
        """
        Calculate comprehensive MAXAVAIL constraint metrics for the day.

        Parameters:
        -----------
        dispatch_results : List[DispatchResult]
            Daily dispatch results from DispatchSimulator
        interval_revenues : List[IntervalRevenue]
            Daily interval revenues

        Returns:
        --------
        MAXAVAILMetrics
            Comprehensive MAXAVAIL metrics
        """
        if not dispatch_results:
            return MAXAVAILMetrics()

        # Basic counts
        total_intervals = len(dispatch_results)
        constrained_intervals = sum(1 for r in dispatch_results
                                    if r.constraint_applied and r.constraint_applied != "none")
        constraint_frequency = (constrained_intervals / total_intervals * 100) if total_intervals > 0 else 0

        # Capacity metrics
        total_requested_mw = sum(r.requested_mw for r in dispatch_results)
        total_maxavail_mw = sum(r.maxavail_for_interval or 0.0 for r in dispatch_results)
        total_dispatched_mw = sum(r.dispatched_mw for r in dispatch_results)

        # Efficiency metrics
        maxavail_utilizations = [r.maxavail_utilization for r in dispatch_results
                                 if r.maxavail_utilization is not None]
        avg_maxavail_utilization = np.mean(maxavail_utilizations) if maxavail_utilizations else 0.0

        dispatch_efficiencies = [r.get_dispatch_efficiency() for r in dispatch_results]
        avg_nemde_efficiency = np.mean(dispatch_efficiencies) if dispatch_efficiencies else 0.0

        capacity_withheld_mw = total_requested_mw - total_dispatched_mw

        # Revenue impact
        estimated_revenue_loss = sum(ir.estimated_revenue_loss for ir in interval_revenues)
        total_revenue = sum(ir.net_revenue for ir in interval_revenues)
        constraint_revenue_impact = (estimated_revenue_loss / (total_revenue + estimated_revenue_loss) * 100) if (
                                                                                                                             total_revenue + estimated_revenue_loss) > 0 else 0

        # Constraint type distribution
        constraint_type_counts = {}
        for result in dispatch_results:
            if result.constraint_applied and result.constraint_applied != "none":
                constraint_type = result.constraint_applied
                constraint_type_counts[constraint_type] = constraint_type_counts.get(constraint_type, 0) + 1

        return MAXAVAILMetrics(
            total_intervals=total_intervals,
            constrained_intervals=constrained_intervals,
            constraint_frequency=constraint_frequency,
            total_requested_mw=total_requested_mw,
            total_maxavail_mw=total_maxavail_mw,
            total_dispatched_mw=total_dispatched_mw,
            average_maxavail_utilization=avg_maxavail_utilization,
            average_nemde_efficiency=avg_nemde_efficiency,
            capacity_withheld_mw=capacity_withheld_mw,
            estimated_revenue_loss=estimated_revenue_loss,
            constraint_revenue_impact=constraint_revenue_impact,
            constraint_type_counts=constraint_type_counts
        )

    def _create_zero_revenue_breakdown(self, date: datetime) -> RevenueBreakdown:
        """Create a zero revenue breakdown for days with no dispatch"""
        return RevenueBreakdown(
            energy_revenue=0.0,
            capacity_revenue=0.0,
            ancillary_revenue=0.0,
            total_gross_revenue=0.0,
            transmission_losses=0.0,
            total_net_revenue=0.0,
            revenue_per_mwh=0.0,
            revenue_per_mw_capacity=0.0,
            capacity_factor=0.0,
            utilization_rate=0.0,
            maxavail_metrics=MAXAVAILMetrics(),
            dispatch_efficiency=0.0,
            constraint_adjusted_revenue=0.0
        )

    def get_maxavail_analysis_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive MAXAVAIL constraint analysis summary.

        Returns:
        --------
        Dict[str, Any]
            Summary of MAXAVAIL constraint impacts
        """
        if not self.maxavail_events:
            return {
                'total_events': 0,
                'message': 'No MAXAVAIL constraint events recorded'
            }

        events_df = pd.DataFrame(self.maxavail_events)

        # Constraint frequency analysis
        total_intervals = len(self.maxavail_events)
        constrained_intervals = len(events_df[events_df['constraint_type'] != ConstraintType.NONE])

        # Revenue impact analysis
        total_revenue_loss = events_df['revenue_loss'].sum()
        constraint_types = events_df['constraint_applied'].value_counts()

        # MAXAVAIL utilization analysis
        maxavail_events = events_df[events_df['constraint_applied'] == 'maxavail']
        avg_maxavail_utilization = maxavail_events['maxavail_utilization'].mean() if len(maxavail_events) > 0 else 0

        return {
            'total_events': total_intervals,
            'constrained_intervals': constrained_intervals,
            'constraint_frequency_pct': (constrained_intervals / total_intervals * 100) if total_intervals > 0 else 0,
            'total_revenue_loss': total_revenue_loss,
            'constraint_type_distribution': constraint_types.to_dict(),
            'maxavail_specific_events': len(maxavail_events),
            'average_maxavail_utilization_pct': avg_maxavail_utilization * 100,
            'most_common_constraint': constraint_types.index[0] if len(constraint_types) > 0 else 'none'
        }

    def export_constraint_analysis(self, output_path: str) -> pd.DataFrame:
        """
        Export detailed MAXAVAIL constraint analysis to CSV.

        Parameters:
        -----------
        output_path : str
            Path to save the analysis CSV

        Returns:
        --------
        pd.DataFrame
            MAXAVAIL constraint analysis data
        """
        if not self.maxavail_events:
            self.logger.warning("No MAXAVAIL constraint events to export")
            return pd.DataFrame()

        constraint_df = pd.DataFrame(self.maxavail_events)
        constraint_df.to_csv(output_path, index=False)

        self.logger.info(f"MAXAVAIL constraint analysis exported to: {output_path}")
        return constraint_df

    def export_interval_revenues(self, output_path: str) -> pd.DataFrame:
        """
        Export detailed interval revenue analysis to CSV.

        Parameters:
        -----------
        output_path : str
            Path to save the revenue CSV

        Returns:
        --------
        pd.DataFrame
            Interval revenue analysis data
        """
        if not self.interval_revenues:
            self.logger.warning("No interval revenue data to export")
            return pd.DataFrame()

        # Convert interval revenues to DataFrame
        revenue_data = []
        for ir in self.interval_revenues:
            revenue_data.append({
                'interval_datetime': ir.interval_datetime,
                'interval_number': ir.interval_number,
                'strategy_name': ir.strategy_name,
                'dispatched_mwh': ir.dispatched_mwh,
                'rrp': ir.rrp,
                'gross_revenue': ir.gross_revenue,
                'net_revenue': ir.net_revenue,
                'maxavail': ir.maxavail,
                'requested_capacity': ir.requested_capacity,
                'constraint_applied': ir.constraint_applied,
                'dispatch_efficiency': ir.dispatch_efficiency,
                'revenue_efficiency': ir.revenue_efficiency,
                'estimated_revenue_loss': ir.estimated_revenue_loss
            })

        revenue_df = pd.DataFrame(revenue_data)
        revenue_df.to_csv(output_path, index=False)

        self.logger.info(f"Interval revenue analysis exported to: {output_path}")
        return revenue_df

    def reset_calculator(self):
        """Reset calculator state for new analysis"""
        self.daily_revenues.clear()
        self.interval_revenues.clear()
        self.maxavail_events.clear()
        self.constraint_impact_history.clear()

        self.logger.info("Revenue calculator reset")

    def get_daily_revenues(self) -> Dict[str, float]:
        """Get dictionary of daily revenues by date"""
        return self.daily_revenues.copy()

    def get_total_revenue(self) -> float:
        """Get total revenue across all days"""
        return sum(self.daily_revenues.values())

    def get_constraint_impact_summary(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of constraint impacts, optionally filtered by strategy.

        Parameters:
        -----------
        strategy_name : str, optional
            Filter by specific strategy name

        Returns:
        --------
        Dict[str, Any]
            Constraint impact summary
        """
        revenues = self.interval_revenues
        if strategy_name:
            revenues = [ir for ir in revenues if ir.strategy_name == strategy_name]

        if not revenues:
            return {'message': 'No revenue data available'}

        total_revenue_loss = sum(ir.estimated_revenue_loss for ir in revenues)
        total_revenue = sum(ir.net_revenue for ir in revenues)
        constrained_intervals = sum(1 for ir in revenues if ir.constraint_applied != "none")

        return {
            'total_intervals': len(revenues),
            'constrained_intervals': constrained_intervals,
            'constraint_frequency_pct': (constrained_intervals / len(revenues) * 100),
            'total_revenue_loss': total_revenue_loss,
            'total_revenue_realized': total_revenue,
            'revenue_impact_pct': (total_revenue_loss / (total_revenue + total_revenue_loss) * 100) if (
                                                                                                                   total_revenue + total_revenue_loss) > 0 else 0,
            'average_dispatch_efficiency_pct': np.mean([ir.dispatch_efficiency for ir in revenues]) * 100,
            'average_revenue_efficiency_pct': np.mean([ir.revenue_efficiency for ir in revenues]) * 100
        }