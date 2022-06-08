import pandas as pd

from smartarp.agents.generic_agent import GenericAgent
from smartarp.forecast.improving_forecaster import ImprovingForecaster
from smartarp.agents.microgrid import Microgrid
from smartarp.core.flexibility_bid import BidStatus
from smartarp.utils import bernoulli_distribution

POS = 'pos'
NEG = 'neg'
EPS = 1e-4


class RetailerGeneric(GenericAgent):
    """
    Representation of the retailer.
    """
    def __init__(self):

        self._simulation_period = None

        self.demand_base = None
        self.demand_total = None
        self.individual_demands = None

        self._dam_prices = None

        self.energy_imbalance_pos = None
        self.energy_imbalance_flex_pos = None
        self.energy_imbalance_neg = None
        self.energy_imbalance_flex_neg = None

        self.costs_dam_provision = None
        self.costs_dam_provision_flexible = None
        self.costs_imbalance_pos = None
        self.costs_imbalance_flex_pos = None
        self.costs_imbalance_neg = None
        self.costs_imbalance_flex_neg = None
        self.costs_flexibility = None

        self.initial_baseline = None
        self.initial_baseline_no_microgrids = None
        self.initial_flexible_baseline = None
        self.revenues = None

        self.retail_price = None
        self.deviation_ratio = None

        self.microgrid_baselines = None
        self.microgrid_initial_baselines = None
        self.microgrid_consumption = None
        self.microgrid_retailing_costs = None
        self.microgrid_deviation_pos = None
        self.microgrid_deviation_neg = None
        self.microgrid_deviation_costs_pos = None
        self.microgrid_deviation_costs_neg = None

        self.forecaster = None
        self.microgrids = None
        self.flexibility_bids = None  # Set of flexibility bids.
        self.available_flexibility_RT_upward = None
        self.available_flexibility_RT_downward = None
        self.available_flexibility_upward = None
        self.available_flexibility_downward = None
        self.dam_flexibility_upward = None
        self.dam_flexibility_downward = None

        self.use_dam_flexibility = None
        self.use_rt_flexibility = None

        # Economic analysis
        self.purchasing_price_global = None
        self.purchasing_price_local = None
        self.selling_price_global = None
        self.selling_price_local = None
        self.distribution_price_global = None
        self.distribution_price_local = None
        self.taxes_price = None
        self.retailer_mode = None

    def initialise(self, configuration):
        self.demand_base = pd.read_csv(
            configuration.path_demand_retailer,
            header=0,
            index_col=0,
            usecols=[0, 1],
            parse_dates=True,
            infer_datetime_format=True,
            squeeze=True
        ) / configuration.demand_conversion_to_MWh

        # SImulation period
        self._simulation_period = configuration.simulation_period()

        # Use of flexibility
        self.use_dam_flexibility = configuration.use_dam_flexibility
        self.use_rt_flexibility = configuration.use_rt_flexibility

        self.retail_price = configuration.retail_price
        self.deviation_ratio = configuration.deviation_ratio

        self.demand_total = pd.Series(data=0.0, index=self._simulation_period)

        self._dam_prices = pd.Series(data=0.0, index=self._simulation_period)

        self.energy_imbalance_pos = pd.Series(data=0.0, index=self._simulation_period)
        self.energy_imbalance_flex_pos = pd.Series(data=0.0, index=self._simulation_period)
        self.energy_imbalance_neg = pd.Series(data=0.0, index=self._simulation_period)
        self.energy_imbalance_flex_neg = pd.Series(data=0.0, index=self._simulation_period)

        self.costs_dam_provision = pd.Series(data=0.0, index=self._simulation_period)
        self.costs_dam_provision_flexible = pd.Series(data=0.0, index=self._simulation_period)
        self.costs_imbalance_pos = pd.Series(data=0.0, index=self._simulation_period)
        self.costs_imbalance_flex_pos = pd.Series(data=0.0, index=self._simulation_period)
        self.costs_imbalance_neg = pd.Series(data=0.0, index=self._simulation_period)
        self.costs_imbalance_flex_neg = pd.Series(data=0.0, index=self._simulation_period)
        self.costs_flexibility = pd.Series(data=0.0, index=self._simulation_period)

        self.initial_baseline = pd.Series(data=0.0, index=self._simulation_period)
        self.initial_baseline_no_microgrids = pd.Series(data=0.0, index=self._simulation_period)
        self.initial_flexible_baseline = pd.Series(data=0.0, index=self._simulation_period)
        self.dam_flexibility_upward = pd.Series(data=0.0, index=self._simulation_period)
        self.dam_flexibility_downward = pd.Series(data=0.0, index=self._simulation_period)

        self.revenues = pd.Series(data=0.0, index=self._simulation_period)

        self.forecaster = ImprovingForecaster()
        self.forecaster.train(self.demand_base)

        self.flexibility_bids = set()

        # Create microgrids
        self.individual_demands = pd.read_csv(
            configuration.path_demand_amr, header=0, index_col=0, parse_dates=True, infer_datetime_format=True,
            squeeze=True
        ) / configuration.demand_conversion_to_MWh

        self.available_flexibility_RT_upward = dict()
        self.available_flexibility_RT_downward = dict()
        self.available_flexibility_upward = dict()
        self.available_flexibility_downward = dict()
        self.microgrid_baselines = dict()
        self.microgrid_initial_baselines = dict()
        self.microgrid_consumption = dict()
        self.microgrid_retailing_costs = dict()
        self.microgrid_deviation_pos = dict()
        self.microgrid_deviation_neg = dict()
        self.microgrid_deviation_costs_pos = dict()
        self.microgrid_deviation_costs_neg = dict()
        self.microgrids = list()
        for microgrid_name, microgrid_parameters in configuration.microgrids.items():
            m = Microgrid(microgrid_name, self.individual_demands[microgrid_name], **microgrid_parameters)
            m.initialise(configuration)
            self.microgrids.append(m)

            # Initialise 0 baselines for each microgrid
            self.microgrid_baselines[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.microgrid_initial_baselines[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.microgrid_consumption[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.microgrid_retailing_costs[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.microgrid_deviation_pos[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.microgrid_deviation_neg[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.microgrid_deviation_costs_pos[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.microgrid_deviation_costs_neg[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.available_flexibility_RT_upward[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.available_flexibility_RT_downward[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.available_flexibility_upward[m] = pd.Series(data=0.0, index=self._simulation_period)
            self.available_flexibility_downward[m] = pd.Series(data=0.0, index=self._simulation_period)

        # Initialise baseline_forecast
        self._compute_initial_baseline(configuration, now=self._simulation_period[0] - pd.Timedelta(hours=12))

        # Initialise economic analysis
        self.purchasing_price_global = configuration.purchasing_price_global
        self.purchasing_price_local = configuration.purchasing_price_local
        self.selling_price_global = configuration.selling_price_global
        self.selling_price_local = configuration.selling_price_local
        self.distribution_price_global = configuration.distribution_price_global
        self.distribution_price_local = configuration.distribution_price_local
        self.taxes_price = configuration.taxes_price
        self.retailer_mode = configuration.retailer_mode

    def act(self, now: pd.Timestamp, system):
        # Remove revoked or expired flexibility bids
        expired = [
            b for b in self.flexibility_bids if (b.status == BidStatus.REVOKED or b.check_expiration(now))
        ]
        self.flexibility_bids.difference_update(expired)

        # Realisation of the retailer and microgrid actions
        microgrid_consumption = 0.0
        for m in self.microgrids:
            m.act(now, system)
            self.microgrid_consumption[m][now] = m.consumption(now)
            microgrid_consumption += self.microgrid_consumption[m][now]
        self.demand_total[now] = self.demand_base[now] + microgrid_consumption

        # Revenues retailer
        self.revenues[now] = self.demand_total[now] * self.retail_price

        # DAM baseline_forecast including flexibility in day-ahead
        if now.hour == 12 and now.minute == 0:
            self._compute_initial_baseline(system.configuration, now=now)

        # Activation of flexibility bids
        self.bid_activation(now, system)

        # Compute DAM costs
        time_improved_forecast = None  # TODO: To use for improved forecast
        self.costs_dam_provision[now] = self.initial_baseline[now] * system.dam.price(now, time_improved_forecast) # TODO: forecast only for optimisation.
        self.costs_dam_provision_flexible[now] = self.initial_flexible_baseline[now] * system.dam.price(now, time_improved_forecast)
        # TODO: check dam provision costs for flexible demand

        # Imbalance
        time_forecast = None
        imbalance_prices_pos = system.tso.price(now, time_forecast)[POS]
        imbalance_prices_neg = system.tso.price(now, time_forecast)[NEG]
        energy_imbalance = self.demand_total[now] - self.initial_baseline[now]  # Positive when short
        energy_imbalance_flexible = self.demand_total[now] - self.initial_flexible_baseline[now]  # Positive when short
        # TODO: imbalance costs for flexible baseline

        if energy_imbalance > 0:
            self.energy_imbalance_neg[now] = energy_imbalance
            self.energy_imbalance_pos[now] = 0.0

        else:
            self.energy_imbalance_neg[now] = 0.0
            self.energy_imbalance_pos[now] = energy_imbalance

        if energy_imbalance_flexible > 0:
            self.energy_imbalance_flex_neg[now] = energy_imbalance_flexible
            self.energy_imbalance_flex_pos[now] = 0.0

        else:
            self.energy_imbalance_flex_neg[now] = 0.0
            self.energy_imbalance_flex_pos[now] = energy_imbalance_flexible

        # Imbalance costs
        self.costs_imbalance_neg[now] = self.energy_imbalance_neg[now] * imbalance_prices_pos
        self.costs_imbalance_flex_neg[now] = self.energy_imbalance_flex_neg[now] * imbalance_prices_pos
        self.costs_imbalance_pos[now] = self.energy_imbalance_pos[now] * imbalance_prices_neg
        self.costs_imbalance_flex_pos[now] = self.energy_imbalance_flex_pos[now] * imbalance_prices_neg

        # Deviation microgrids
        for m in self.microgrids:
            microgrid_deviation = self.microgrid_consumption[m][now] - m.schedule[now]

            if microgrid_deviation > 0:
                self.microgrid_deviation_neg[m][now] = microgrid_deviation
                self.microgrid_deviation_pos[m][now] = 0.0

                self.microgrid_deviation_costs_neg[m][now] = microgrid_deviation * imbalance_prices_pos * \
                                                             self.deviation_ratio
                self.microgrid_deviation_costs_pos[m][now] = 0.0

            else:
                self.microgrid_deviation_neg[m][now] = 0.0
                self.microgrid_deviation_pos[m][now] = microgrid_deviation

                self.microgrid_deviation_costs_neg[m][now] = 0.0
                self.microgrid_deviation_costs_pos[m][now] = microgrid_deviation * imbalance_prices_neg * \
                                                             self.deviation_ratio

    def bid_activation(self, now: pd.Timestamp, system):
        """
        Activates the required bids.
        """
        if self.use_rt_flexibility:
            self._rt_activation(system, now, time_end=system.configuration.simulation_period()[-1],
                                resolution=system.configuration.resolution,
                                resolution_int=system.configuration.resolution_int)

        last_dam_period = system.configuration.simulation_period()[-1]
        last_dam_day = pd.Timestamp(year=last_dam_period.year, month=last_dam_period.month, day=last_dam_period.day)
        if self.use_dam_flexibility:
            if (
                    (now.hour == 12)
                    and (now.minute == 0)
                    and (pd.Timestamp(year=now.year, month=now.month, day=now.day) < last_dam_day)
            ):
                self._dam_activation(system, now, time_end=system.configuration.simulation_period()[-1])

    def _dam_activation(self, system, now: pd.Timestamp, time_end: pd.Timestamp):
        """
        Activates bids in day ahead market
        """
        next_day = pd.Timestamp(year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo) + pd.Timedelta(days=1)
        for time in pd.date_range(next_day, next_day + pd.Timedelta(days=1), freq=system.configuration.resolution_str(),
                                  closed='left'):

            # Retrieve flexibility bids from the day-ahead
            last_time_step = time_end - pd.Timedelta(minutes=system.configuration.resolution)
            if time <= last_time_step:
                self.initial_flexible_baseline[time] += self.initial_baseline[time]

                for m in self.microgrids:
                    self.flexibility_bids.update(m.flexibility(system, time_stamp=time, now=now))

                for b in self.flexibility_bids:
                    # Filter bids not starting at the given time stamp
                    if b.flexibility.index[0] != time or b.status != BidStatus.FREE:
                        continue

                    # Processing the bid
                    b.status = BidStatus.PENDING

                    if b.status == BidStatus.PENDING:
                        # Reserving the bid
                        b.status = BidStatus.RESERVED

                        if bernoulli_distribution(.2):
                            b.status = BidStatus.ACCEPTED
                            b.acceptance = 1
                            self._compute_flexibility_costs(time, b)

                            for t, v in b.flexibility.iteritems():
                                self.initial_flexible_baseline[t] += v

                        else:
                            b.status = BidStatus.REJECTED
                    else:
                        b.status = BidStatus.FREE

    def _rt_activation(self, system, now: pd.Timestamp, time_end: pd.Timestamp):
        """
        Activates bids in real time
        """
        time = now + pd.Timedelta(minutes=15)

        # Computation of the retailer's imbalance
        forecasted_imbalance = self.forecaster.forecast(time, now) + \
                               sum(self.microgrid_baselines[m][time] for m in self.microgrids) - \
                               self.initial_baseline[time]  # Positive when short

        remaining_imbalance = forecasted_imbalance
        # Checking the flexibility bids for possible matches
        for b in self.flexibility_bids:
            # Filter bids not starting at the given time stamp
            if b.flexibility.index[0] != time or b.status != BidStatus.FREE:
                continue

            # Processing the bid
            b.status = BidStatus.PENDING

            if not (-EPS <= remaining_imbalance <= EPS):
                # Reserving the bid
                b.status = BidStatus.RESERVED

                if remaining_imbalance > 0.0:
                    b.acceptance = max(0.0, min(1.0, remaining_imbalance / b.flexibility[0]))

                if b.acceptance is None or b.acceptance < EPS:
                    b.status = BidStatus.REJECTED
                else:
                    b.status = BidStatus.ACCEPTED
                    remaining_imbalance -= b.flexibility[0] * b.acceptance
                    self._compute_flexibility_costs(time, b)
            else:
                b.status = BidStatus.FREE

    def _compute_initial_baseline(self, configuration, now: pd.Timestamp):
        """
        Computes the initial baseline of the retailer and the microgrid offering flexibility.

        :param configuration: System configuration
        :param now: Time stamp of the simulation process.
        """
        # Create the baseline_forecast of the retailer
        next_day = pd.Timestamp(year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo) + pd.Timedelta(days=1)
        for time in pd.date_range(next_day, next_day + pd.Timedelta(days=1), freq=configuration.resolution_str(),
                                  closed='left'):
            self.initial_baseline[time] = self.forecaster.forecast(time, now)
            self.initial_baseline_no_microgrids[time] = self.initial_baseline[time]
            for m in self.microgrids:
                # Get the forecast from the microgrid
                microgrid_baseline = m.baseline_forecast(time, now)

                # Accept it
                m.set_schedule(time, microgrid_baseline)
                self.microgrid_initial_baselines[m][time] = microgrid_baseline
                self.microgrid_baselines[m][time] = microgrid_baseline

                # Add it to the total
                self.initial_baseline[time] += microgrid_baseline
                self.initial_flexible_baseline[time] = self.initial_baseline[time]

    def _compute_flexibility_costs(self, time_stamp: pd.Timestamp, bid: pd.Series):
        """
        Computes the costs of accepting the flexibility bids
        """

        bid_volume = abs(bid.flexibility[0])
        bid_cost = bid.cost
        bid_acceptance = bid.acceptance

        self.costs_flexibility[time_stamp] = bid_volume * bid_cost * bid_acceptance
