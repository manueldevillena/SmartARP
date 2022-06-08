import pandas as pd

from smartarp.agents.generic_agent import GenericAgent

class TSO(GenericAgent):
    """
    Representation of the Transmission System Operator (TSO).
    """
    def __init__(self):
        self.imbalance_prices = None

    def initialise(self, configuration):
        self.imbalance_prices = pd.read_csv(configuration.path_imbalance_prices,
                                  header=0,
                                  index_col=0,
                                  usecols=[0, 3, 4],
                                  parse_dates=True,
                                  infer_datetime_format=True,
                                  squeeze=True)

    def act(self, time_stamp: pd.Timestamp, system):
        pass

    def price(self, time_stamp: pd.Timestamp, now: pd.Timestamp): # TODO include the forecast
        return self.imbalance_prices.loc[time_stamp]
