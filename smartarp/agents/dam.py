import pandas as pd

from smartarp.agents.generic_agent import GenericAgent
from smartarp.forecast.shift_forecaster import ShiftForecaster
from smartarp.utils import closest


class DAM(GenericAgent):
    """
    Representation of the day-ahead market.
    """
    def __init__(self):
        self.prices = None
        self._simulation_period = None

        # Time series
        self.price_forecast = None

        # Pick the forecaster
        self.forecaster = ShiftForecaster()

    def initialise(self, configuration):
        self.prices = pd.read_csv(configuration.path_dam_prices,
                                  header=0,
                                  index_col=0,
                                  usecols=[0, 1],
                                  parse_dates=True,
                                  infer_datetime_format=True,
                                  squeeze=True)

        # Retrieve simulation period
        self._simulation_period = configuration.simulation_period_hourly()

        if self.forecaster is not None:
            # Initialise forecaster
            self.forecaster.train(self.prices)

        # Price forecast
        self.price_forecast = pd.Series(data=0.0, index=self._simulation_period)

    def act(self, now: pd.Timestamp, system):
        pass

    def _price_forecast(self, now: pd.Timestamp):
        """
        Forecast of the day-ahead market price.

        :param now: Current time stamp.
        """
        # Forecast day-ahead market prices
        for time in self._simulation_period:
            self.price_forecast[time] = self.forecaster.forecast(time, now)

    def price(self, time_stamp: pd.Timestamp, now: pd.Timestamp, forecast: bool = False):
        """
        Retrieves the day-ahead market price at a given time.

        :param time_stamp: Target time stamp.
        :param now: Current time stamp.
        :param forecast: Boolean to activate the forecast of prices or to simply retrieve actual prices.
        :return: Day-ahead market price.
        """
        if forecast:
            self._price_forecast(now)
            return closest(self.price_forecast, time_stamp)
        else:
            return closest(self.prices, time_stamp)
