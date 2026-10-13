"""pf — multi-factor stock selection and portfolio backtesting.

Pipeline: loader → factors (ret calls knn) → scoring → cluster → selection → weights → backtest → metrics;
config holds parameter defaults and path constants.
"""
