"""
Advanced Logging System for Battery Bidding Backtest System
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import traceback


class LogLevel(Enum):
    """Enhanced log levels for battery backtesting"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    PERFORMANCE = "PERFORMANCE"  # Custom level for performance metrics
    REVENUE = "REVENUE"  # Custom level for revenue events
    DISPATCH = "DISPATCH"  # Custom level for dispatch events


@dataclass
class LogEvent:
    """Structured log event for analysis"""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    strategy: Optional[str] = None
    revenue: Optional[float] = None
    dispatch_mw: Optional[float] = None
    rrp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceTimer:
    """Context manager for performance timing"""

    def __init__(self, logger: logging.Logger, operation_name: str, threshold_seconds: float = 1.0):
        self.logger = logger
        self.operation_name = operation_name
        self.threshold_seconds = threshold_seconds
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        if duration > self.threshold_seconds:
            self.logger.log(logging.INFO + 5, f"{self.operation_name} completed in {duration:.2f}s")
        else:
            self.logger.debug(f"{self.operation_name} completed in {duration:.2f}s")

        if exc_type is not None:
            self.logger.error(f"{self.operation_name} failed after {duration:.2f}s: {exc_val}")


class BatteryBacktestLogger:
    """
    Advanced logging system specifically designed for battery backtesting.

    Features:
    - Multiple log files for different purposes
    - Structured logging for analysis
    - Performance monitoring
    - Revenue and dispatch event tracking
    - Log aggregation and analysis
    - Automatic log rotation
    """

    def __init__(self, config, base_name: str = "battery_backtest"):
        """
        Initialize the advanced logging system.

        Parameters:
        -----------
        config : Configuration
            Configuration object containing output directory
        base_name : str
            Base name for log files
        """
        self.config = config
        self.base_name = base_name
        self.log_dir = Path(config.output_directory) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Add custom log levels
        self._add_custom_levels()

        # Event tracking
        self.log_events: List[LogEvent] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.revenue_events: List[Dict[str, Any]] = []
        self.dispatch_events: List[Dict[str, Any]] = []

        # Initialize loggers
        self.loggers = self._setup_loggers()

        # Main logger
        self.logger = self.loggers['main']

        self.logger.info("Advanced logging system initialized")

    def _add_custom_levels(self):
        """Add custom logging levels"""
        # Performance level (between INFO and WARNING)
        logging.addLevelName(25, "PERFORMANCE")
        logging.addLevelName(22, "REVENUE")
        logging.addLevelName(21, "DISPATCH")

    def _setup_loggers(self) -> Dict[str, logging.Logger]:
        """Setup multiple specialized loggers"""
        loggers = {}

        # Main application logger
        loggers['main'] = self._create_logger(
            'battery_backtest.main',
            self.log_dir / f"{self.base_name}_main.log",
            level=logging.INFO
        )

        # Performance logger
        loggers['performance'] = self._create_logger(
            'battery_backtest.performance',
            self.log_dir / f"{self.base_name}_performance.log",
            level=logging.DEBUG,
            format_string='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )

        # Revenue logger
        loggers['revenue'] = self._create_logger(
            'battery_backtest.revenue',
            self.log_dir / f"{self.base_name}_revenue.log",
            level=logging.INFO
        )

        # Dispatch logger
        loggers['dispatch'] = self._create_logger(
            'battery_backtest.dispatch',
            self.log_dir / f"{self.base_name}_dispatch.log",
            level=logging.INFO
        )

        # Error logger (errors only)
        loggers['error'] = self._create_logger(
            'battery_backtest.error',
            self.log_dir / f"{self.base_name}_errors.log",
            level=logging.ERROR
        )

        # Debug logger (everything)
        loggers['debug'] = self._create_logger(
            'battery_backtest.debug',
            self.log_dir / f"{self.base_name}_debug.log",
            level=logging.DEBUG
        )

        return loggers

    def _create_logger(self, name: str, file_path: Path,
                      level: int = logging.INFO,
                      format_string: Optional[str] = None) -> logging.Logger:
        """Create a specialized logger with file and console handlers"""

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter

        # Clear existing handlers
        logger.handlers.clear()

        # Default format
        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        formatter = logging.Formatter(format_string)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler for main logger only
        if name == 'battery_backtest.main':
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Custom handler for event tracking
        event_handler = EventTrackingHandler(self)
        event_handler.setLevel(logging.DEBUG)
        logger.addHandler(event_handler)

        return logger

    def log_performance(self, operation: str, duration: float, metadata: Optional[Dict] = None):
        """Log performance metrics"""
        self.loggers['performance'].log(25, f"PERF: {operation} took {duration:.3f}s",
                                       extra={'metadata': metadata or {}})

        # Track for analysis
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        self.performance_metrics[operation].append(duration)

    def log_revenue(self, strategy: str, amount: float, interval: int,
                   rrp: float, dispatched_mw: float, metadata: Optional[Dict] = None):
        """Log revenue events"""
        message = f"REVENUE: {strategy} earned ${amount:.2f} (Interval {interval}, {dispatched_mw:.1f}MW @ ${rrp:.2f})"
        self.loggers['revenue'].log(22, message)

        # Track for analysis
        revenue_event = {
            'timestamp': datetime.now(),
            'strategy': strategy,
            'amount': amount,
            'interval': interval,
            'rrp': rrp,
            'dispatched_mw': dispatched_mw,
            'metadata': metadata or {}
        }
        self.revenue_events.append(revenue_event)

    def log_dispatch(self, strategy: str, interval: int, requested_mw: float,
                    dispatched_mw: float, rrp: float, status: str,
                    metadata: Optional[Dict] = None):
        """Log dispatch events"""
        efficiency = (dispatched_mw / requested_mw * 100) if requested_mw > 0 else 100
        message = f"DISPATCH: {strategy} I{interval} - Requested: {requested_mw:.1f}MW, " \
                 f"Dispatched: {dispatched_mw:.1f}MW ({efficiency:.1f}%) @ ${rrp:.2f} [{status}]"

        self.loggers['dispatch'].log(21, message)

        # Track for analysis
        dispatch_event = {
            'timestamp': datetime.now(),
            'strategy': strategy,
            'interval': interval,
            'requested_mw': requested_mw,
            'dispatched_mw': dispatched_mw,
            'rrp': rrp,
            'status': status,
            'efficiency': efficiency,
            'metadata': metadata or {}
        }
        self.dispatch_events.append(dispatch_event)

    def log_error(self, error: Exception, context: str = "", metadata: Optional[Dict] = None):
        """Log errors with full context"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            'metadata': metadata or {}
        }

        message = f"ERROR in {context}: {type(error).__name__}: {str(error)}"
        self.loggers['error'].error(message, extra={'error_info': error_info})

    def timer(self, operation_name: str, threshold_seconds: float = 1.0) -> PerformanceTimer:
        """Create a performance timer context manager"""
        return PerformanceTimer(self.loggers['performance'], operation_name, threshold_seconds)

    def log_strategy_summary(self, strategy: str, summary_data: Dict[str, Any]):
        """Log strategy performance summary"""
        self.logger.info(f"Strategy {strategy.upper()} Summary:")
        self.logger.info(f"  Total Revenue: ${summary_data.get('total_revenue', 0):.2f}")
        self.logger.info(f"  Total Dispatched: {summary_data.get('total_dispatched_mwh', 0):.1f} MWh")
        self.logger.info(f"  Dispatch Events: {summary_data.get('dispatch_events', 0)}")
        self.logger.info(f"  Avg Revenue/MWh: ${summary_data.get('avg_revenue_per_mwh', 0):.2f}")
        self.logger.info(f"  Capacity Utilization: {summary_data.get('capacity_utilization', 0):.1f}%")

    def log_daily_summary(self, date: datetime, market_conditions: Dict[str, Any],
                         best_strategy: str, best_revenue: float):
        """Log daily market summary"""
        self.logger.info(f"Daily Summary for {date.date()}:")
        self.logger.info(f"  Average RRP: ${market_conditions.get('avg_rrp', 0):.2f}")
        self.logger.info(f"  Max RRP: ${market_conditions.get('max_rrp', 0):.2f}")
        self.logger.info(f"  Min RRP: ${market_conditions.get('min_rrp', 0):.2f}")
        self.logger.info(f"  Price Volatility: ${market_conditions.get('price_volatility', 0):.2f}")
        self.logger.info(f"  High Price Intervals (>$100): {market_conditions.get('high_price_intervals', 0)}")
        self.logger.info(f"  Total Intervals: {market_conditions.get('total_intervals', 0)}")
        self.logger.info(f"  Best Strategy: {best_strategy} (Revenue: ${best_revenue:.2f})")

    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze logged performance metrics"""
        analysis = {}

        for operation, durations in self.performance_metrics.items():
            if durations:
                analysis[operation] = {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'avg_time': sum(durations) / len(durations),
                    'min_time': min(durations),
                    'max_time': max(durations),
                    'operations_per_second': len(durations) / sum(durations) if sum(durations) > 0 else 0
                }

        return analysis

    def get_revenue_analysis(self) -> Dict[str, Any]:
        """Analyze logged revenue events"""
        if not self.revenue_events:
            return {'message': 'No revenue events logged'}

        total_revenue = sum(event['amount'] for event in self.revenue_events)
        total_dispatched = sum(event['dispatched_mw'] / 12 for event in self.revenue_events)  # Convert to MWh

        # Strategy breakdown
        strategy_revenues = {}
        for event in self.revenue_events:
            strategy = event['strategy']
            if strategy not in strategy_revenues:
                strategy_revenues[strategy] = {'revenue': 0, 'events': 0, 'dispatched_mwh': 0}

            strategy_revenues[strategy]['revenue'] += event['amount']
            strategy_revenues[strategy]['events'] += 1
            strategy_revenues[strategy]['dispatched_mwh'] += event['dispatched_mw'] / 12

        # Calculate averages
        for strategy_data in strategy_revenues.values():
            if strategy_data['dispatched_mwh'] > 0:
                strategy_data['avg_revenue_per_mwh'] = strategy_data['revenue'] / strategy_data['dispatched_mwh']
            else:
                strategy_data['avg_revenue_per_mwh'] = 0

        return {
            'total_revenue': total_revenue,
            'total_dispatched_mwh': total_dispatched,
            'total_events': len(self.revenue_events),
            'avg_revenue_per_mwh': total_revenue / total_dispatched if total_dispatched > 0 else 0,
            'strategy_breakdown': strategy_revenues
        }

    def get_dispatch_analysis(self) -> Dict[str, Any]:
        """Analyze logged dispatch events"""
        if not self.dispatch_events:
            return {'message': 'No dispatch events logged'}

        total_requested = sum(event['requested_mw'] for event in self.dispatch_events)
        total_dispatched = sum(event['dispatched_mw'] for event in self.dispatch_events)

        # Status breakdown
        status_counts = {}
        for event in self.dispatch_events:
            status = event['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        # Strategy efficiency
        strategy_efficiency = {}
        for event in self.dispatch_events:
            strategy = event['strategy']
            if strategy not in strategy_efficiency:
                strategy_efficiency[strategy] = {'requested': 0, 'dispatched': 0, 'events': 0}

            strategy_efficiency[strategy]['requested'] += event['requested_mw']
            strategy_efficiency[strategy]['dispatched'] += event['dispatched_mw']
            strategy_efficiency[strategy]['events'] += 1

        # Calculate efficiency percentages
        for strategy_data in strategy_efficiency.values():
            if strategy_data['requested'] > 0:
                strategy_data['efficiency'] = strategy_data['dispatched'] / strategy_data['requested'] * 100
            else:
                strategy_data['efficiency'] = 100

        return {
            'total_events': len(self.dispatch_events),
            'total_requested_mw': total_requested,
            'total_dispatched_mw': total_dispatched,
            'overall_efficiency': (total_dispatched / total_requested * 100) if total_requested > 0 else 100,
            'status_breakdown': status_counts,
            'strategy_efficiency': strategy_efficiency
        }

    def export_log_analysis(self, output_path: str):
        """Export comprehensive log analysis to JSON"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'performance_analysis': self.get_performance_analysis(),
            'revenue_analysis': self.get_revenue_analysis(),
            'dispatch_analysis': self.get_dispatch_analysis(),
            'log_summary': {
                'total_log_events': len(self.log_events),
                'revenue_events': len(self.revenue_events),
                'dispatch_events': len(self.dispatch_events),
                'performance_operations': len(self.performance_metrics)
            }
        }

        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        self.logger.info(f"Log analysis exported to: {output_path}")
        return analysis

    def create_performance_report(self) -> str:
        """Create a formatted performance report"""
        perf_analysis = self.get_performance_analysis()
        revenue_analysis = self.get_revenue_analysis()
        dispatch_analysis = self.get_dispatch_analysis()

        report = []
        report.append("=" * 60)
        report.append("BATTERY BACKTEST PERFORMANCE REPORT")
        report.append("=" * 60)

        # Performance metrics
        if perf_analysis:
            report.append("\nPERFORMANCE METRICS:")
            report.append("-" * 30)
            for operation, metrics in perf_analysis.items():
                report.append(f"{operation}:")
                report.append(f"  Count: {metrics['count']}")
                report.append(f"  Total Time: {metrics['total_time']:.2f}s")
                report.append(f"  Average Time: {metrics['avg_time']:.3f}s")
                report.append(f"  Operations/Second: {metrics['operations_per_second']:.2f}")

        # Revenue summary
        if isinstance(revenue_analysis, dict) and 'total_revenue' in revenue_analysis:
            report.append("\nREVENUE SUMMARY:")
            report.append("-" * 30)
            report.append(f"Total Revenue: ${revenue_analysis['total_revenue']:.2f}")
            report.append(f"Total Energy: {revenue_analysis['total_dispatched_mwh']:.1f} MWh")
            report.append(f"Avg Revenue/MWh: ${revenue_analysis['avg_revenue_per_mwh']:.2f}")
            report.append(f"Revenue Events: {revenue_analysis['total_events']}")

        # Dispatch summary
        if isinstance(dispatch_analysis, dict) and 'total_events' in dispatch_analysis:
            report.append("\nDISPATCH SUMMARY:")
            report.append("-" * 30)
            report.append(f"Total Events: {dispatch_analysis['total_events']}")
            report.append(f"Overall Efficiency: {dispatch_analysis['overall_efficiency']:.1f}%")
            report.append(f"Total Requested: {dispatch_analysis['total_requested_mw']:.1f} MW")
            report.append(f"Total Dispatched: {dispatch_analysis['total_dispatched_mw']:.1f} MW")

        report.append("=" * 60)

        return "\n".join(report)

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        for log_file in self.log_dir.glob("*.log*"):
            try:
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime < cutoff_date:
                    log_file.unlink()
                    self.logger.info(f"Deleted old log file: {log_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to delete old log file {log_file.name}: {str(e)}")

    def close(self):
        """Close all loggers and handlers"""
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

        self.logger.info("Logging system closed")


class EventTrackingHandler(logging.Handler):
    """Custom logging handler to track events for analysis"""

    def __init__(self, backtest_logger: BatteryBacktestLogger):
        super().__init__()
        self.backtest_logger = backtest_logger

    def emit(self, record: logging.LogRecord):
        """Capture log record for event tracking"""
        try:
            # Create structured log event
            log_event = LogEvent(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                module=record.module,
                function=record.funcName,
                line_number=record.lineno,
                metadata=getattr(record, 'metadata', None)
            )

            # Extract strategy, revenue, and dispatch info if available
            message = record.getMessage()
            if 'REVENUE:' in message and hasattr(record, 'metadata'):
                # Extract revenue information from log message
                pass  # Revenue events are tracked separately

            self.backtest_logger.log_events.append(log_event)

        except Exception:
            # Don't let logging errors break the application
            pass


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Parameters:
    -----------
    name : str
        Name of the logger (usually __name__)

    Returns:
    --------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)