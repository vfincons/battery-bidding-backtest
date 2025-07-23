import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class BiddingStrategy(Enum):
    """Available bidding strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    PEAK_CAPTURE = "peak_capture"


@dataclass
class BiddingProfile:
    """Container for a complete bidding profile with fixed MW allocations"""
    name: str
    band_allocations: Dict[str, float]  # PRICEBAND -> Fixed MW allocation
    total_capacity: float
    remaining_soc: float
    current_interval: int
    minimum_floor_mw: float = 5.0  # Minimum stable operating point
    last_override_applied: Optional[str] = None  # ADD THIS LINE

    def get_remaining_capacity(self) -> float:
        """Calculate total remaining capacity across all bands"""
        return sum(self.band_allocations.values())

    def is_exhausted(self) -> bool:
        """Check if all capacity is allocated"""
        return self.remaining_soc <= 0.01  # Small tolerance for floating point

    def is_below_minimum_floor(self) -> bool:
        """Check if remaining SOC is below minimum stable operating point"""
        return self.remaining_soc < self.minimum_floor_mw


class BiddingStrategyManager:
    """
    Manages bidding strategies with fixed MW allocation per price band.

    KEY CHANGES FROM ORIGINAL:
    - Initial allocation creates fixed MW amounts per band (not proportions)
    - No dynamic reallocation during peak period
    - 5MW minimum floor replaces final override mechanism
    - Band allocations decrease only when capacity is dispatched
    """

    def __init__(self, total_capacity: float = 100.0, total_intervals: int = 31,
                 minimum_floor_mw: float = 5.0, max_ramp_per_interval: float = 16.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize BiddingStrategyManager with proper parameter validation.
        FIXED: Added logger parameter and proper validation.
        """
        self.total_capacity = total_capacity
        self.total_intervals = total_intervals
        self.minimum_floor_mw = minimum_floor_mw
        self.max_ramp_per_interval = max_ramp_per_interval  # ADDED for interface alignment
        self.final_override_threshold = 4  # Results from May2024 - Aug2024 imply 4
        self.logger = logger or logging.getLogger(__name__)

        # Validate parameters
        if total_capacity <= 0:
            raise ValueError("total_capacity must be positive")
        if total_intervals <= 0:
            raise ValueError("total_intervals must be positive")
        if minimum_floor_mw < 0:
            raise ValueError("minimum_floor_mw must be non-negative")
        if minimum_floor_mw >= total_capacity:
            raise ValueError("minimum_floor_mw must be less than total_capacity")
        if max_ramp_per_interval <= 0:
            raise ValueError("max_ramp_per_interval must be positive")

        # Strategy proportions - used ONLY for initial allocation
        self.strategy_proportions = {
            BiddingStrategy.CONSERVATIVE: {
                'PRICEBAND1': 0.32, 'PRICEBAND2': 0.23, 'PRICEBAND3': 0.18, 'PRICEBAND4': 0.13,
                'PRICEBAND5': 0.08, 'PRICEBAND6': 0.06, 'PRICEBAND7': 0.00, 'PRICEBAND8': 0.00,
                'PRICEBAND9': 0.00, 'PRICEBAND10': 0.00
            },
            BiddingStrategy.BALANCED: {
                'PRICEBAND1': 0.10, 'PRICEBAND2': 0.18, 'PRICEBAND3': 0.17, 'PRICEBAND4': 0.15,
                'PRICEBAND5': 0.13, 'PRICEBAND6': 0.10, 'PRICEBAND7': 0.08, 'PRICEBAND8': 0.05,
                'PRICEBAND9': 0.03, 'PRICEBAND10': 0.01
            },
            BiddingStrategy.AGGRESSIVE: {
                'PRICEBAND1': 0.03, 'PRICEBAND2': 0.08, 'PRICEBAND3': 0.09, 'PRICEBAND4': 0.10,
                'PRICEBAND5': 0.11, 'PRICEBAND6': 0.12, 'PRICEBAND7': 0.13, 'PRICEBAND8': 0.15,
                'PRICEBAND9': 0.18, 'PRICEBAND10': 0.01
            },
            BiddingStrategy.PEAK_CAPTURE: {
                'PRICEBAND1': 0.00, 'PRICEBAND2': 0.00, 'PRICEBAND3': 0.00, 'PRICEBAND4': 0.00,
                'PRICEBAND5': 0.00, 'PRICEBAND6': 0.00, 'PRICEBAND7': 0.00, 'PRICEBAND8': 0.43,
                'PRICEBAND9': 0.57, 'PRICEBAND10': 0.00
            }
        }

        # Validate proportions sum to 1.0
        self._validate_proportions()

        # Current profiles for active strategies
        self.current_profiles = {}
        self.dispatch_history = []

        # MAXAVAIL tracking for validation
        self.maxavail_history = []

        self.logger.debug(f"BiddingStrategyManager initialized: capacity={total_capacity}MW, "
                          f"intervals={total_intervals}, min_floor={minimum_floor_mw}MW")

    def _validate_proportions(self) -> None:
        """Validate that all strategy proportions sum to 1.0"""
        for strategy, proportions in self.strategy_proportions.items():
            total = sum(proportions.values())
            if abs(total - 1.0) > 0.001:  # Allow small floating point tolerance
                raise ValueError(f"Strategy {strategy.value} proportions sum to {total}, not 1.0")

    def initialize_strategy(self, strategy: BiddingStrategy, current_interval: int = 1) -> BiddingProfile:
        """
        Initialize a bidding strategy with FIXED MW allocations per price band.

        CHANGED: Creates fixed MW amounts that remain static during peak period.

        Args:
            strategy: The bidding strategy to initialize
            current_interval: Current dispatch interval (1-31)

        Returns:
            BiddingProfile: Initialized profile with fixed MW allocations
        """
        if strategy not in self.strategy_proportions:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Convert proportions to FIXED MW allocations
        proportions = self.strategy_proportions[strategy]
        band_allocations = {}

        for priceband, proportion in proportions.items():
            band_allocations[priceband] = proportion * self.total_capacity

        # Create profile with fixed allocations
        profile = BiddingProfile(
            name=strategy.value,
            band_allocations=band_allocations,
            total_capacity=self.total_capacity,
            remaining_soc=self.total_capacity,
            current_interval=current_interval,
            minimum_floor_mw=self.minimum_floor_mw
        )

        # Store as current profile
        self.current_profiles[strategy.value] = profile

        self.logger.info(f"Initialized strategy {strategy.value} with fixed MW allocations:")
        for band, allocation in band_allocations.items():
            if allocation > 0:
                self.logger.info(f"  {band}: {allocation:.1f}MW")

        return profile

    def get_bid_schedule(self, strategy_name: str, current_interval: int) -> Dict[str, float]:
        """
        Get the current bidding schedule with fixed allocations, 5MW minimum floor check,
        and 7-interval final override.

        FIXED: Proper fixed allocation logic - NO reallocation of remaining SOC.
        """
        if strategy_name not in self.current_profiles:
            raise ValueError(f"Strategy '{strategy_name}' not initialized")

        profile = self.current_profiles[strategy_name]
        profile.current_interval = current_interval
        profile.last_override_applied = None

        # Check 1: 5MW minimum floor check
        if profile.is_below_minimum_floor():
            self.logger.debug(f"5MW minimum floor triggered for {strategy_name} at interval {current_interval}")
            return self._apply_minimum_floor(profile)

        # Check 2: 7-interval final override check logic
        intervals_remaining = self.total_intervals - current_interval + 1
        if intervals_remaining <= self.final_override_threshold and profile.remaining_soc > 0:
            self.logger.debug(f"7-interval final override triggered for {strategy_name} at interval {current_interval}")
            return self._apply_final_override(profile)

        # FIXED: Return CURRENT band allocations without reallocation
        # This is the key fix - we return the current state, not recalculated proportions
        return profile.band_allocations.copy()

    def _apply_final_override(self, profile: BiddingProfile) -> Dict[str, float]:
        """
        Apply 7-interval final override mechanism: move all remaining capacity to PRICEBAND1.

        RETAINED: Original final override logic alongside 5MW minimum floor.

        Args:
            profile: Current bidding profile

        Returns:
            Dict mapping PRICEBAND to MW allocation (all in PRICEBAND1)
        """
        # Create empty allocation
        allocations = {f'PRICEBAND{i}': 0.0 for i in range(1, 11)}

        # Move all remaining capacity to PRICEBAND1 (Guaranteed tier)
        total_remaining = sum(profile.band_allocations.values())
        allocations['PRICEBAND1'] = total_remaining

        # Update profile allocations
        profile.band_allocations = allocations.copy()
        profile.last_override_applied = "final_override"  # ADD THIS LINE

        self.logger.debug(f"7-interval final override applied for {profile.name}: "
                          f"{total_remaining:.2f}MW moved to PRICEBAND1")

        return allocations

    def _apply_minimum_floor(self, profile: BiddingProfile) -> Dict[str, float]:
        """
        Apply 5MW minimum floor mechanism: move all remaining capacity to PRICEBAND1.

        NEW: 5MW minimum floor logic alongside final override mechanism.

        Args:
            profile: Current bidding profile

        Returns:
            Dict mapping PRICEBAND to MW allocation (all in PRICEBAND1)
        """
        # Create empty allocation
        allocations = {f'PRICEBAND{i}': 0.0 for i in range(1, 11)}

        # Move all remaining capacity to PRICEBAND1 (Guaranteed tier)
        total_remaining = sum(profile.band_allocations.values())
        allocations['PRICEBAND1'] = total_remaining

        # Update profile allocations
        profile.band_allocations = allocations.copy()
        profile.last_override_applied = "minimum_floor"

        self.logger.debug(f"5MW minimum floor applied for {profile.name}: "
                          f"{total_remaining:.2f}MW moved to PRICEBAND1")

        return allocations
    def get_bid_schedule_with_maxavail_check(self, strategy_name: str, interval_number: int,
                                             maxavail: float) -> Dict[str, float]:
        """
        Get a bid schedule that represents true strategy allocation.

        FIXED: No longer scales BANDAVAIL to fit MAXAVAIL - MAXAVAIL is a constraint
        applied by NEMDE algorithm, not a scaling factor for bid submission.

        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        interval_number : int
            Current interval number
        maxavail : float
            Maximum available capacity for the interval (constraint, not scaling factor)

        Returns:
        --------
        Dict[str, float]
            True BANDAVAIL allocation (strategy allocation, not scaled)
        """
        # Get base allocation from strategy (fixed allocations)
        base_allocation = self.get_bid_schedule(strategy_name, interval_number)

        # Calculate total BANDAVAIL for tracking purposes only
        total_bandavail = sum(base_allocation.values())

        # Track MAXAVAIL event for analysis (but don't scale)
        self._track_maxavail_event(strategy_name, interval_number, maxavail, total_bandavail)

        # CRITICAL FIX: Return true strategy allocation, not scaled
        # MAXAVAIL constraint is applied by NEMDE algorithm in DispatchSimulator
        self.logger.debug(f"Bid schedule for {strategy_name} interval {interval_number}: "
                          f"Total BANDAVAIL={total_bandavail:.2f}MW, MAXAVAIL={maxavail:.2f}MW "
                          f"(NEMDE will apply MAXAVAIL constraint)")

        return base_allocation

    def _track_maxavail_event(self, strategy_name: str, interval_number: int,
                              maxavail: float, total_bandavail: float) -> None:
        """Track MAXAVAIL events for analysis and validation."""
        maxavail_event = {
            'strategy': strategy_name,
            'interval': interval_number,
            'maxavail': maxavail,
            'total_bandavail': total_bandavail,
            'scaling_required': total_bandavail > maxavail,
            'scaling_factor': maxavail / total_bandavail if total_bandavail > 0 else 0
        }
        self.maxavail_history.append(maxavail_event)

    def update_after_dispatch(self, strategy_name: str, dispatched_mw: float,
                              current_interval: int) -> bool:
        """
        Update strategy state after dispatch occurs by reducing SOC only.

        FIXED: This method now only updates SOC. Band allocation updates are handled
        separately by update_band_allocations_after_nemde_dispatch().

        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        dispatched_mw : float
            Total amount dispatched
        current_interval : int
            Current interval number

        Returns:
        --------
        bool
            True if capacity remains, False if exhausted
        """
        if strategy_name not in self.current_profiles:
            raise ValueError(f"Strategy '{strategy_name}' not initialized")

        profile = self.current_profiles[strategy_name]

        # Record pre-dispatch SOC for validation
        pre_dispatch_soc = profile.remaining_soc

        # Update remaining SOC first
        profile.remaining_soc = max(0, profile.remaining_soc - dispatched_mw)
        profile.current_interval = current_interval

        # Record dispatch event for tracking
        dispatch_event = {
            'strategy': strategy_name,
            'interval': current_interval,
            'dispatched_mw': dispatched_mw,
            'remaining_soc_before': pre_dispatch_soc,
            'remaining_soc_after': profile.remaining_soc
        }

        self.dispatch_history.append(dispatch_event)

        self.logger.debug(f"Updated {strategy_name} SOC after dispatch: "
                          f"{dispatched_mw:.2f}MW dispatched, "
                          f"{profile.remaining_soc:.2f}MW remaining")

        # NOTE: Band allocations are updated separately by update_band_allocations_after_nemde_dispatch()
        # This separation ensures proper tracking of which specific bands were dispatched

        return profile.remaining_soc > 0.01

    def update_strategy_after_dispatch(self, strategy_name: str, dispatched_mw: float,
                                       nemde_breakdown: Dict[str, float], current_interval: int) -> bool:
        """
        ATOMIC update of both SOC and band allocations to maintain consistency.

        NEW METHOD: Replaces separate update_after_dispatch and update_band_allocations_after_nemde_dispatch
        """
        if strategy_name not in self.current_profiles:
            raise ValueError(f"Strategy '{strategy_name}' not initialized")

        profile = self.current_profiles[strategy_name]

        # Record pre-update state for validation
        pre_dispatch_soc = profile.remaining_soc
        pre_dispatch_allocations = profile.band_allocations.copy()

        self.logger.debug(f"ATOMIC UPDATE - {strategy_name} Interval {current_interval}:")
        self.logger.debug(f"  Pre-dispatch SOC: {pre_dispatch_soc:.2f}MW")
        self.logger.debug(f"  Total dispatched: {dispatched_mw:.2f}MW")
        self.logger.debug(f"  NEMDE breakdown: {nemde_breakdown}")

        # STEP 1: Update remaining SOC
        profile.remaining_soc = max(0, profile.remaining_soc - dispatched_mw)
        profile.current_interval = current_interval

        # STEP 2: Update band allocations based on NEMDE breakdown
        total_reduction = 0.0
        for band_name, dispatched_amount in nemde_breakdown.items():
            if dispatched_amount > 0.001 and band_name in profile.band_allocations:
                original_allocation = profile.band_allocations[band_name]

                # CRITICAL: Ensure we don't reduce below zero
                reduction = min(dispatched_amount, original_allocation)
                new_allocation = max(0, original_allocation - reduction)
                profile.band_allocations[band_name] = new_allocation
                total_reduction += reduction

                self.logger.debug(f"  {band_name}: {original_allocation:.2f}MW → {new_allocation:.2f}MW "
                                  f"(reduced: {reduction:.2f}MW)")

        # STEP 3: VALIDATION - Check consistency
        post_dispatch_allocation_sum = sum(profile.band_allocations.values())
        expected_allocation_sum = profile.remaining_soc

        consistency_error = abs(post_dispatch_allocation_sum - expected_allocation_sum)

        if consistency_error > 0.01:
            self.logger.error(f"ATOMIC UPDATE CONSISTENCY ERROR - {strategy_name}:")
            self.logger.error(f"  Remaining SOC: {profile.remaining_soc:.3f}MW")
            self.logger.error(f"  Sum of allocations: {post_dispatch_allocation_sum:.3f}MW")
            self.logger.error(f"  Difference: {consistency_error:.3f}MW")
            self.logger.error(f"  Total reduction applied: {total_reduction:.3f}MW")
            self.logger.error(f"  Expected reduction: {dispatched_mw:.3f}MW")

            # RECOVERY: Force consistency by proportional scaling
            if post_dispatch_allocation_sum > 0 and profile.remaining_soc >= 0:
                scaling_factor = profile.remaining_soc / post_dispatch_allocation_sum
                self.logger.warning(f"  Applying recovery scaling factor: {scaling_factor:.6f}")

                for band_name in profile.band_allocations:
                    profile.band_allocations[band_name] *= scaling_factor

        else:
            self.logger.debug(f"  ✅ Consistency check passed: difference = {consistency_error:.6f}MW")

        # STEP 4: Record dispatch event
        dispatch_event = {
            'strategy': strategy_name,
            'interval': current_interval,
            'dispatched_mw': dispatched_mw,
            'nemde_breakdown': nemde_breakdown.copy(),
            'pre_dispatch_soc': pre_dispatch_soc,
            'post_dispatch_soc': profile.remaining_soc,
            'consistency_error': consistency_error,
            'total_reduction_applied': total_reduction
        }
        self.dispatch_history.append(dispatch_event)

        return profile.remaining_soc > 0.01

    def update_band_allocations_after_nemde_dispatch(self, strategy_name: str,
                                                     nemde_dispatch_breakdown: Dict[str, float]) -> None:
        """
        Update band allocations based on NEMDE dispatch breakdown.

        FIXED: This method properly reduces only the bands that were dispatched
        by the exact amounts dispatched from each band.

        This method should be called by the DispatchSimulator after NEMDE dispatch
        to reduce the specific band allocations that were dispatched.

        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        nemde_dispatch_breakdown : Dict[str, float]
            Breakdown of how much was dispatched from each band (from NEMDE result)
        """
        if strategy_name not in self.current_profiles:
            raise ValueError(f"Strategy '{strategy_name}' not initialized")

        profile = self.current_profiles[strategy_name]

        self.logger.debug(f"Updating band allocations for {strategy_name} after NEMDE dispatch")

        # Reduce band allocations based on actual NEMDE dispatch breakdown
        total_reduction = 0.0

        for band_name, dispatched_amount in nemde_dispatch_breakdown.items():
            if dispatched_amount > 0.001 and band_name in profile.band_allocations:
                # Get current allocation for this band
                original_allocation = profile.band_allocations[band_name]

                # Reduce this band's allocation by the dispatched amount
                new_allocation = max(0, original_allocation - dispatched_amount)
                profile.band_allocations[band_name] = new_allocation

                total_reduction += dispatched_amount

                self.logger.debug(f"  {band_name}: {original_allocation:.2f}MW → {new_allocation:.2f}MW "
                                  f"(dispatched: {dispatched_amount:.2f}MW)")

        # Verify total reduction matches expected
        if total_reduction > 0.001:
            self.logger.debug(f"Total band allocation reduction: {total_reduction:.2f}MW")

        # Double-check: total allocations should equal remaining SOC
        total_remaining_allocation = sum(profile.band_allocations.values())
        if abs(total_remaining_allocation - profile.remaining_soc) > 0.01:
            self.logger.warning(f"Band allocation total ({total_remaining_allocation:.2f}MW) "
                                f"!= remaining SOC ({profile.remaining_soc:.2f}MW) for {strategy_name}")

    def get_remaining_soc(self, strategy_name: str) -> float:
        """Get the remaining state of charge for a strategy"""
        if strategy_name not in self.current_profiles:
            raise ValueError(f"Strategy '{strategy_name}' not initialized")

        return self.current_profiles[strategy_name].remaining_soc

    def is_final_override_active(self, strategy_name: str) -> bool:
        """
        Check if either minimum floor OR dynamic final override mechanism is currently active.

        FIXED: Now uses dynamic SOC-based final override instead of fixed interval threshold.
        """
        if strategy_name not in self.current_profiles:
            return False

        profile = self.current_profiles[strategy_name]

        # Check 5MW minimum floor first (the highest priority)
        if profile.is_below_minimum_floor():
            self.logger.debug(f"5MW minimum floor active for {strategy_name}")
            return True

        # Dynamic final override based on remaining SOC and discharge capacity
        intervals_remaining = self.total_intervals - profile.current_interval + 1
        remaining_soc = profile.remaining_soc

        # Calculate theoretical discharge capacity in remaining intervals
        theoretical_discharge_capacity = intervals_remaining * self.max_ramp_per_interval

        # Trigger final override if:
        # 1. We're in the final 3 intervals, AND
        # 2. Remaining SOC exceeds what can be naturally discharged
        if intervals_remaining <= 3 and remaining_soc > theoretical_discharge_capacity:
            self.logger.debug(f"Dynamic final override active for {strategy_name}: "
                              f"SOC={remaining_soc:.2f}MW > theoretical_capacity={theoretical_discharge_capacity:.2f}MW "
                              f"with {intervals_remaining} intervals remaining")
            return True

        # Safety net for the very final interval
        if intervals_remaining == 1 and remaining_soc > 5.0:
            self.logger.debug(f"Final interval safety override for {strategy_name}: "
                              f"SOC={remaining_soc:.2f}MW remaining")
            return True

        return False

    def was_final_override_applied(self, strategy_name: str) -> bool:
        """Check if the final override was applied in the last get_bid_schedule call."""
        if strategy_name not in self.current_profiles:
            return False
        return self.current_profiles[strategy_name].last_override_applied == "final_override"

    def was_minimum_floor_applied(self, strategy_name: str) -> bool:
        """Check if the minimum floor was applied in the last get_bid_schedule call."""
        if strategy_name not in self.current_profiles:
            return False
        return self.current_profiles[strategy_name].last_override_applied == "minimum_floor"

    def get_current_bid_schedule(self, strategy_name: str) -> Dict[str, float]:
        """
        Get a current bidding schedule for a strategy (compatibility method).

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dict mapping PRICEBAND to MW allocation
        """
        if strategy_name not in self.current_profiles:
            raise ValueError(f"Strategy '{strategy_name}' not initialized")

        profile = self.current_profiles[strategy_name]
        return self.get_bid_schedule(strategy_name, profile.current_interval)

    def reset_strategy(self, strategy_name: str) -> None:
        """Reset a strategy to its initial state"""
        if strategy_name not in self.current_profiles:
            raise ValueError(f"Strategy '{strategy_name}' not found")

        # Find the corresponding strategy enum
        strategy_enum = BiddingStrategy(strategy_name)

        # Reinitialize the strategy
        self.initialize_strategy(strategy_enum)

    def reset_all_strategies(self) -> None:
        """Reset all strategies to their initial states"""
        self.current_profiles.clear()
        self.dispatch_history.clear()
        self.maxavail_history.clear()

    def get_dispatch_summary(self, strategy_name: str) -> Dict:
        """Get a summary of dispatch history for a strategy"""
        if strategy_name not in self.current_profiles:
            return {'error': f"Strategy '{strategy_name}' not found"}

        profile = self.current_profiles[strategy_name]
        strategy_dispatches = [d for d in self.dispatch_history if d['strategy'] == strategy_name]

        if not strategy_dispatches:
            return {
                'total_dispatched': 0,
                'dispatch_events': 0,
                'remaining_soc': profile.remaining_soc,
                'utilization_rate': 0.0,
                'minimum_floor_used': False
            }

        total_dispatched = sum(d['dispatched_mw'] for d in strategy_dispatches)
        minimum_floor_used = profile.is_below_minimum_floor()

        # Check if final override was used (7-interval mechanism)
        final_override_used = any(
            self.total_intervals - d['interval'] + 1 <= self.final_override_threshold
            for d in strategy_dispatches
        )

        return {
            'total_dispatched': total_dispatched,
            'dispatch_events': len(strategy_dispatches),
            'remaining_soc': profile.remaining_soc,
            'utilization_rate': ((self.total_capacity - profile.remaining_soc) / self.total_capacity) * 100,
            'minimum_floor_used': minimum_floor_used,
            'final_override_used': final_override_used
        }

    def get_maxavail_summary(self, strategy_name: str) -> Dict:
        """
        Get a summary of MAXAVAIL constraint events for a strategy.
        UNCHANGED from the original implementation.
        """
        strategy_maxavail_events = [e for e in self.maxavail_history if e['strategy'] == strategy_name]

        if not strategy_maxavail_events:
            return {
                'total_intervals': 0,
                'scaling_required_count': 0,
                'scaling_frequency': 0.0,
                'average_scaling_factor': 1.0,
                'average_maxavail_utilization': 0.0
            }

        scaling_events = [e for e in strategy_maxavail_events if e['scaling_required']]
        scaling_factors = [e['scaling_factor'] for e in strategy_maxavail_events if e['scaling_required']]
        utilization_rates = [e['maxavail'] / e['total_bandavail']
                             for e in strategy_maxavail_events if e['total_bandavail'] > 0]

        return {
            'total_intervals': len(strategy_maxavail_events),
            'scaling_required_count': len(scaling_events),
            'scaling_frequency': len(scaling_events) / len(strategy_maxavail_events) * 100,
            'average_scaling_factor': np.mean(scaling_factors) if scaling_factors else 1.0,
            'average_maxavail_utilization': np.mean(utilization_rates) if utilization_rates else 0.0,
            'min_scaling_factor': min(scaling_factors) if scaling_factors else 1.0,
            'max_scaling_factor': max(scaling_factors) if scaling_factors else 1.0
        }

    def get_strategy_info(self, strategy_name: str) -> Dict:
        """
        Get detailed information about a strategy.

        CHANGED: Updated to reflect fixed allocation approach.
        """
        if strategy_name not in self.current_profiles:
            return {'error': f"Strategy '{strategy_name}' not found"}

        profile = self.current_profiles[strategy_name]
        strategy_enum = BiddingStrategy(strategy_name)

        # Add MAXAVAIL summary to strategy info
        maxavail_summary = self.get_maxavail_summary(strategy_name)

        return {
            'name': strategy_name,
            'total_capacity': self.total_capacity,
            'remaining_soc': profile.remaining_soc,
            'current_interval': profile.current_interval,
            'minimum_floor_mw': self.minimum_floor_mw,
            'final_override_threshold': self.final_override_threshold,
            'minimum_floor_active': profile.is_below_minimum_floor(),
            'final_override_active': (
                        self.total_intervals - profile.current_interval + 1 <= self.final_override_threshold and profile.remaining_soc > 0),
            'original_proportions': self.strategy_proportions[strategy_enum].copy(),
            'current_band_allocations': profile.band_allocations.copy(),
            'maxavail_summary': maxavail_summary
        }

    def validate_bandavail_consistency(self, allocation: Dict[str, float], maxavail: float) -> Dict[str, Any]:
        """
        Validate that BANDAVAIL allocation is consistent with MAXAVAIL constraint.
        UNCHANGED from the original implementation.
        """
        total_bandavail = sum(allocation.values())

        validation_result = {
            'is_valid': total_bandavail <= maxavail + 0.001,
            'total_bandavail': total_bandavail,
            'maxavail': maxavail,
            'excess_amount': max(0, total_bandavail - maxavail),
            'scaling_factor_needed': maxavail / total_bandavail if total_bandavail > 0 else 0,
            'utilization_rate': total_bandavail / maxavail if maxavail > 0 else 0
        }

        if not validation_result['is_valid']:
            self.logger.warning(f"BANDAVAIL consistency check failed: "
                                f"Total BANDAVAIL ({total_bandavail:.2f}MW) exceeds "
                                f"MAXAVAIL ({maxavail:.2f}MW)")

        return validation_result

    def update_capacity_after_dispatch(self, strategy_name: str, dispatched_mw: float,
                                       dispatched_bands: List[str]) -> bool:
        """
        Update strategy capacity after dispatch occurs.
        INTERFACE METHOD: Required by DispatchSimulator for backward compatibility.

        Args:
            strategy_name: Name of the strategy
            dispatched_mw: Amount of MW dispatched
            dispatched_bands: List of price bands that were dispatched (for analysis)

        Returns:
            bool: True if capacity remains, False if exhausted
        """
        if strategy_name not in self.current_profiles:
            raise ValueError(f"Strategy '{strategy_name}' not initialized")

        profile = self.current_profiles[strategy_name]

        # Record dispatch event with band information
        dispatch_event = {
            'strategy': strategy_name,
            'interval': profile.current_interval,
            'dispatched_mw': dispatched_mw,
            'dispatched_bands': dispatched_bands,  # For analysis
            'remaining_soc_before': profile.remaining_soc,
            'remaining_soc_after': max(0, profile.remaining_soc - dispatched_mw)
        }

        # Use the main update method
        result = self.update_after_dispatch(strategy_name, dispatched_mw, profile.current_interval)

        # Update dispatch event record
        dispatch_event['remaining_soc_after'] = profile.remaining_soc
        self.dispatch_history.append(dispatch_event)

        return result

    def generate_bidding_profiles(self, matrix_df: pd.DataFrame) -> Dict[str, 'BiddingProfile']:
        """Compatibility method - generates profiles for all strategies"""
        profiles = {}
        for strategy_enum in BiddingStrategy:
            profile = self.initialize_strategy(strategy_enum)
            profiles[strategy_enum.value] = profile
        return profiles

    def get_remaining_capacity(self, strategy_name: str) -> float:
        """
        Get remaining capacity for a strategy.
        INTERFACE METHOD: Required by DispatchSimulator for compatibility.

        Args:
            strategy_name: Name of the strategy

        Returns:
            float: Remaining capacity in MW
        """
        return self.get_remaining_soc(strategy_name)

    def export_maxavail_analysis(self, output_path: str) -> pd.DataFrame:
        """Export MAXAVAIL constraint analysis to CSV - UNCHANGED"""
        if not self.maxavail_history:
            self.logger.warning("No MAXAVAIL history to export")
            return pd.DataFrame()

        maxavail_df = pd.DataFrame(self.maxavail_history)
        maxavail_df.to_csv(output_path, index=False)

        self.logger.info(f"MAXAVAIL analysis exported to: {output_path}")
        return maxavail_df
# # Example usage and testing
# if __name__ == "__main__":
#     # Example matrix data (you would load this from your actual data)
#     sample_matrix = pd.DataFrame({
#         'PRICEBAND': range(1, 11),
#         'Actual_Probability': [60, 45, 30, 20, 15, 10, 8, 5, 3, 1],
#         'Expected_Revenue_When_Dispatched': [50, 75, 100, 150, 200, 300, 500, 800, 1200, 2000],
#         'RRP_Threshold': [25, 50, 75, 100, 150, 200, 300, 500, 800, 1500]
#     })
#
#     # Initialize strategy manager
#     strategy_manager = BiddingStrategyManager(total_capacity=100.0)
#
#     # Generate profiles
#     profiles = strategy_manager.generate_bidding_profiles(sample_matrix)
#
#     # Display initial allocations
#     print("Initial Bidding Profiles:")
#     for name, profile in profiles.items():
#         print(f"\n{name} Strategy:")
#         for band, allocation in profile.allocations.items():
#             print(f"  {band}: {allocation:.2f} MW")
#         print(f"  Total: {profile.get_remaining_capacity():.2f} MW")
#
#     # Simulate dispatch events
#     print("\n" + "=" * 50)
#     print("DISPATCH SIMULATION")
#     print("=" * 50)
#
#     strategy_name = 'Balanced'
#     print(f"\nTesting {strategy_name} strategy:")
#
#     # Simulate dispatch event 1
#     dispatched_mw = 15.0
#     dispatched_bands = ['PRICEBAND1', 'PRICEBAND2']
#
#     print(f"\nDispatch Event 1: {dispatched_mw} MW from bands {dispatched_bands}")
#     print(f"Remaining before: {strategy_manager.get_remaining_capacity(strategy_name):.2f} MW")
#
#     strategy_manager.update_capacity_after_dispatch(strategy_name, dispatched_mw, dispatched_bands)
#
#     print(f"Remaining after: {strategy_manager.get_remaining_capacity(strategy_name):.2f} MW")
#     print("Updated allocations:")
#     for band, allocation in strategy_manager.get_current_bid_schedule(strategy_name).items():
#         print(f"  {band}: {allocation:.2f} MW")
#
#     # Get dispatch summary
#     summary = strategy_manager.get_dispatch_summary(strategy_name)
#     print(f"\nDispatch Summary for {strategy_name}:")
#     print(f"  Total dispatched: {summary['total_dispatched']:.2f} MW")
#     print(f"  Dispatch events: {summary['dispatch_events']}")
#     print(f"  Remaining capacity: {summary['remaining_capacity']:.2f} MW")
#     print(f"  Utilization rate: {summary['utilization_rate']:.1f}%")