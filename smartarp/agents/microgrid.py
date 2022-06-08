from typing import List

import pandas as pd

from smartarp.agents.generic_agent import GenericAgent
from smartarp.forecast.shift_forecaster import ShiftForecaster
from smartarp.core.flexibility_bid import FlexibilityBid, BidStatus
from smartarp.utils import closest


class Microgrid(GenericAgent):
    """
    Representation of the aggregation of microgrids.
    """

    def __init__(self, name: str, time_series: pd.Series, relative_payback: float = 1.0, payback_duration: int = 60,
                 idle_time: int = 120, relative_flexibility: float = .2, flexibility_margin: float = 0.1,
                 flexibility_provider: bool = False, producer: bool = False):
        """
        InConstructor.

        :param name: Name of the microgrid.
        :param time_series: Original demand of the microgrid.
        :param relative_payback: How much of the flexibility offered has to be returned.
        :param payback_duration: Horizon in minutes of the payback after providing flexibility.
        :param relative_flexibility: Percentage of flexibility to provide.
        :param flexibility_margin: Relative amount of flexibility conserved by the microgrid for its own purposes and
        self balancing.
        """
        self.forecaster = None
        self.flexibility_bids = set()  #: Set of flexibility bids sent to the retailer.

        # Parameters
        self.name = name
        self.flexibility_provider = flexibility_provider
        self.producer = producer
        self._relative_payback = relative_payback
        self._payback_duration = pd.Timedelta(minutes=payback_duration)
        self._idle_time = pd.Timedelta(minutes=idle_time)
        self._relative_flexibility = relative_flexibility
        self._flexibility_margin = flexibility_margin
        self._costs = None
        self._conversion_factor = None
        self.bids_number = 0

        # Time series
        self.forecaster = ShiftForecaster()
        self.initial_demand = time_series
        self.demand = time_series.copy()
        self.schedule = None
        self.flex_pos = None  #: Positive flexibility sold time series.
        self.flex_neg = None  #: Negative flexibility sold time series.
        self.production = None
        # Simulation period
        self._simulation_period = None

    def initialise(self, configuration):
        # Initialise time series
        self._simulation_period = configuration.simulation_period()
        self.schedule = pd.Series(index=self._simulation_period)
        self.flex_pos = pd.Series(data=0.0, index=self._simulation_period)
        self.flex_neg = pd.Series(data=0.0, index=self._simulation_period)

        # Initialise paremeters
        self._costs = configuration.cost_per_bid

        # Initialise forecaster
        self.forecaster.train(self.initial_demand)

    def act(self, now: pd.Timestamp, system):
        # Clean flexibility
        to_remove = set()
        for b in self.flexibility_bids:
            if b.status == BidStatus.EXPIRED:
                to_remove.add(b)
                continue

            try:
                b.status = BidStatus.REVOKED
                to_remove.add(b)
            except ValueError:
                pass  # The bid cannot be revoked

            if b.status == BidStatus.ACCEPTED:
                self.bids_number += 1
                for t, v in b.flexibility.iteritems():
                    if t <= self._simulation_period[-1]:
                        v_accepted = v * b.acceptance
                        if v_accepted >= 0.0:
                            self.flex_pos[t] += v_accepted
                        else:
                            self.flex_neg[t] += v_accepted
                to_remove.add(b)
        self.flexibility_bids.difference_update(to_remove)

    def consumption(self, time_stamp: pd.Timestamp) -> float:
        """
        Retrieves the microgrid consumption for a given time stamp.

        :param time_stamp: Target time stamp.
        :return:
        """
        if self.flexibility_provider:
            return max(0.0, closest(self.demand, time_stamp) + self.flex_pos[time_stamp] + self.flex_neg[time_stamp])
        else:
            return closest(self.demand, time_stamp)

    def baseline_forecast(self, time_stamp: pd.Timestamp, now: pd.Timestamp):
        """
        Forecast of the baseline of the microgrid.

        :param time_stamp: Target time stamp.
        :param now: Current time stamp.
        :return: Baseline at time stamp.
        """
        return self.forecaster.forecast(time_stamp, now)

    def set_schedule(self, time_stamp: pd.Timestamp, value: float):
        """
        Set the schedule of the microgrid in a time stamp.
        This method is meant to be called by the retailer.

        :param time_stamp: Target time stamp.
        :param value: New schedule value.
        """
        self.schedule[time_stamp] = value

    def flexibility(self, system, time_stamp: pd.Timestamp, now: pd.Timestamp, market: str = 'rt') -> List[FlexibilityBid]:
        """
        Get the flexibility bids (upward and downward) for a time stamp.

        :param system.
        :param time_stamp: Target time stamp.
        :param now: Current time stamp.
        :param market: Indicates the market in which this flexibility bid is offered.
        :return: List of flexibility bids.
        """
        times = pd.date_range(
            start=time_stamp,
            end=time_stamp + self._idle_time + self._payback_duration + pd.Timedelta(minutes=system.configuration.resolution),
            freq=system.configuration.resolution_str(), tz='UTC',
            closed='left'
        )
        flexibilities = []

        idle_steps = int(self._idle_time.seconds / (system.configuration.resolution * 60))
        payback_steps = len(times) - idle_steps - 1

        # Upward flexibility
        v = self._max_positive_flexibility(time_stamp, now)
        original_payback = -v * self._relative_payback / payback_steps

        # Ensure the payback will not lead to negative consumption
        time_steps_payback = times[-payback_steps:]
        payback_relative_demand = []
        for time in time_steps_payback:
            payback_relative_demand.append(original_payback + closest(self.initial_demand, time))
        min_payback = min(payback_relative_demand)
        if min_payback < 0:
            payback = original_payback - min_payback
            v = -payback * payback_steps / self._relative_payback
        else:
            payback = original_payback
        if v > 0.0:
            flex = pd.Series(data=[v] + [0]*idle_steps + [payback]*payback_steps, index=times)
            flexibilities.append(FlexibilityBid(flexibility=flex, cost=(self._costs * abs(flex[0])), market=market,
                                                owner=self))

        # Downward flexibility
        v = self._min_negative_flexibility(time_stamp, now)
        if abs(v) > closest(self.initial_demand, time_stamp):
            v = closest(self.initial_demand, time_stamp)
        payback = -v * self._relative_payback / payback_steps
        if v < 0.0:
            flex = pd.Series(data=[v] + [0]*idle_steps + [payback]*payback_steps, index=times)
            flexibilities.append(FlexibilityBid(flexibility=flex, cost=(self._costs * abs(flex[0])), market=market,
                                                owner=self))

        self.flexibility_bids.update(flexibilities)

        return flexibilities

    def _max_positive_flexibility(self, time_stamp: pd.Timestamp, now: pd.Timestamp) -> float:
        """
        Get the absolute flexibility available in one time stamp.

        :param time_stamp: Target time stamp.
        :param now: Current time stamp.
        :return: Flexibility as a positive number.
        """
        max_consumption = closest(self.initial_demand, time_stamp) * (1.0 + self._relative_flexibility)
        # TODO subtract flexibility bids
        # TODO internal baseline update could be subtracted
        flex_aux = (max_consumption - self.schedule[time_stamp]) * (1.0 - self._flexibility_margin)
        return max(0.0, flex_aux)

    def _min_negative_flexibility(self, time_stamp: pd.Timestamp, now: pd.Timestamp) -> float:
        """
        Get the absolute flexibility available in one time stamp.

        :param time_stamp: Target time stamp.
        :param now: Current time stamp.
        :return: Flexibility as a negative number.
        """
        min_consumption = closest(self.initial_demand, time_stamp) * (1.0 - self._relative_flexibility)
        # TODO subtract flexibility bids
        # TODO internal baseline update could be subtracted
        flex_aux = (min_consumption - self.schedule[time_stamp]) * (1.0 - self._flexibility_margin)
        return min(0.0, flex_aux)
