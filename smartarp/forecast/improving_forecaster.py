import pandas as pd

from smartarp.forecast.forecaster import Forecaster


class ImprovingForecaster(Forecaster):
    """
    Creates simple forecasts by repeating the value from the previous day
    """

    def __init__(self, time_delta: pd.Timedelta = pd.Timedelta(hours=24), min_error_ratio: float = 0.1):
        """
        Constructor.

        :param time_delta: Time series shift time delta.
        :param min_error_ratio: Minimum ratio of the error that is taken even for an error in t for t.
        """

        self.history = None
        self.time_delta = time_delta
        self.min_error_ratio = min_error_ratio

    def train(self, history: pd.Series):
        self.history = history

    def forecast(self, time_stamp: pd.Timestamp, current_time_stamp: pd.Timestamp) -> float:
        initial_forecast = self.history.loc[time_stamp - self.time_delta]

        realization = self.history[time_stamp]
        error = realization - initial_forecast

        time_ratio = (time_stamp - current_time_stamp).total_seconds() / self.time_delta.total_seconds()
        return realization - error * max(self.min_error_ratio, min(time_ratio, 1.0))
