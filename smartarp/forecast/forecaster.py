from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    """
    Abstract forecast generator.
    """

    def train(self, history: pd.Series):
        """
        Trains the forecasting method with past data

        """
        pass


    @abstractmethod
    def forecast(self, forecasted_times_stamp: pd.Timestamp, now: pd.Timestamp) -> float:
        """
        Forecast a value at the given timestamp.

        :param forecasted_times_stamp: Time stamp to forecast.
        :param now: Current time stamp.
        :return: Forecasted value.
        """
        raise NotImplementedError
