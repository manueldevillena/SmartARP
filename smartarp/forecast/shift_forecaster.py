import pandas as pd

from smartarp.forecast.forecaster import Forecaster


class ShiftForecaster(Forecaster):
    """
    Creates simple forecasts by repeating the value from the previous day
    """

    def __init__(self, time_delta: pd.Timedelta = pd.Timedelta(hours=24)):
        self.history = None
        self.time_delta = time_delta

    def train(self, history: pd.Series):
        self.history = history

    def forecast(self, forecasted_times_stamp: pd.Timestamp, now: pd.Timestamp):
        return self.history.loc[forecasted_times_stamp - self.time_delta]
