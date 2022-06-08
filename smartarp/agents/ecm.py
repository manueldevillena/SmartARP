import numpy as np
import pandas as pd

from pyomo.environ import *
from pyomo.core import ConcreteModel, NonNegativeReals, Objective, Constraint, minimize
from pyomo.opt import SolverFactory

from smartarp.agents.retailer_generic import RetailerGeneric
from smartarp.core.flexibility_bid import BidStatus

POS = 'pos'
NEG = 'neg'
EPS = 1e-4


class ECM(RetailerGeneric):
    """
    Representation of the retailer.
    """

    def _dam_activation(self, system, now: pd.Timestamp, time_end: pd.Timestamp):
        """
        Activates bids in day ahead market

        :param system: System.
        :param now: Current time step.
        :param time_end: End of the next day.
        """

        # Create date range of interest
        next_day = pd.Timestamp(year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo) + pd.Timedelta(days=1)
        date_range_of_interest = pd.date_range(
            next_day,
            next_day + pd.Timedelta(days=1),
            freq=system.configuration.resolution_str(),
            closed='left'
        )

        # Loop over the date range of interest to perform some actions
        for time in date_range_of_interest:
            # Retrieve flexibility bids from the day-ahead
            last_time_step = time_end - pd.Timedelta(minutes=system.configuration.resolution)
            if time <= last_time_step:
                for m in self.microgrids:
                    if m.flexibility_provider:
                        # Create and store flexibility bids
                        self.flexibility_bids.update(m.flexibility(system, time_stamp=time, now=now, market='dam'))

        # Compute the rebound effect
        rebound_periods = max([(len(b.flexibility)) for b in self.flexibility_bids])
        date_range_of_interest_no_rebound = date_range_of_interest[:-rebound_periods]

        # Create dictionaries with demand and production of the microgrids
        demand_ts = pd.Series(data=0.0, index=date_range_of_interest_no_rebound)
        production_ts = pd.Series(data=0.0, index=date_range_of_interest_no_rebound)
        for time in date_range_of_interest_no_rebound:
            for m in self.microgrids:
                value = m.schedule[time]
                if value > 0:
                    demand_ts[time] += value
                else:
                    production_ts[time] += abs(value)
            demand_ts[time] += self.initial_baseline_no_microgrids[time]

        # Create dictionaries with actual consumption and production
        demand_ts_actual = pd.Series(data=0.0, index=date_range_of_interest_no_rebound)
        production_ts_actual = pd.Series(data=0.0, index=date_range_of_interest_no_rebound)
        for time in date_range_of_interest_no_rebound:
            for m in self.microgrids:
                value = m.consumption(time)
                if value >= 0:
                    demand_ts_actual[time] += value
                else:
                    production_ts_actual[time] += abs(value)
            # demand_ts_actual[time] += self.initial_baseline_no_microgrids[time]
            demand_ts_actual[time] += self.demand_base[time]
            # TODO: introduce actual demand without microgrids instead of baseline.

        # Retrieve the bids starting during the date range of interest and labeled as day-ahead market
        flexibility_dam = [b for b in self.flexibility_bids
                           if (b.flexibility.index[0] in date_range_of_interest_no_rebound) and (b.market == "dam")]

        # Optimisation

        # Economic parameters
        purchasing_price_global = self.purchasing_price_global
        purchasing_price_local = self.purchasing_price_local
        selling_price_global = self.selling_price_global
        selling_price_local = self.selling_price_local
        dist_price_global = self.distribution_price_global
        dist_price_local = dist_price_global * self.distribution_price_local
        taxes = self.taxes_price
        # DAM prices
        if self.retailer_mode:
            time_improved_forecast = None  # TODO: To use for improved forecast
            dam_prices = pd.Series(data=0.0, index=date_range_of_interest_no_rebound)
            for t in date_range_of_interest_no_rebound:
                dam_prices[t] += system.dam.price(t, time_improved_forecast, forecast=False)
        else:
            dam_prices = pd.Series(data=purchasing_price_global, index=date_range_of_interest_no_rebound)

        # OLD FORMULATION
        model = ConcreteModel()

        # Create sets
        model.TIME = Set(initialize=[t for t in date_range_of_interest_no_rebound])
        model.BIDS = Set(initialize=[b for b in flexibility_dam])

        # Create parameters
        model.Energy_price_local = Param(model.TIME, initialize=purchasing_price_local)
        model.Dist_price_global = Param(model.TIME, initialize=dist_price_global)
        model.Dist_price_local = Param(model.TIME, initialize=dist_price_local)
        model.Selling_price_local = Param(model.TIME, initialize=selling_price_local)
        model.Selling_price_global = Param(model.TIME, initialize=selling_price_global)

        # Create Variables
        model.x = Var(model.BIDS, within=NonNegativeReals, bounds=(0, 1))
        model.imports = Var(model.TIME, within=NonNegativeReals)
        model.exports = Var(model.TIME, within=NonNegativeReals)
        model.local_consumption = Var(model.TIME, within=NonNegativeReals)
        model.local_production = Var(model.TIME, within=NonNegativeReals)
        model.global_costs = Var(within=NonNegativeReals)
        model.local_costs = Var(within=NonNegativeReals)
        model.bid_costs = Var(within=NonNegativeReals)
        model.global_sales = Var(within=NonNegativeReals)
        model.local_sales = Var(within=NonNegativeReals)

        def _objective(model):
            """
            Objective function to minimise total costs.
            """
            return (
                    model.global_costs + model.local_costs + model.bid_costs
                    - model.global_sales - model.local_sales
            )

        def _global_costs(model):
            """
            Compute dam provision costs.
            """
            return model.global_costs >= sum(
                model.imports[t] * (dam_prices[t] + model.Dist_price_global[t] + taxes)
                for t in model.TIME
            )

        def _local_costs(model):
            """
            Compute network costs associated with energy imports from the main grid.
            """
            return model.local_costs >= sum(
                model.local_consumption[t] * (model.Energy_price_local[t] + model.Dist_price_local[t] + taxes)
                for t in model.TIME
            )

        def _bid_costs(model):
            """
            Compute costs of activating flexibility.
            """
            return model.bid_costs >= sum(
                model.x[b] * b.cost
                for b in model.BIDS
            )

        def _global_sales(model):
            """
            Compute network costs associated with energy transactions within the energy community.
            """
            return model.global_sales <= sum(
                model.exports[t] * model.Selling_price_global[t]
                for t in model.TIME
            )

        def _local_sales(model):
            """
            Compute network costs associated with energy transactions within the energy community.
            """
            return model.local_sales <= sum(
                model.local_production[t] * model.Selling_price_local[t]
                for t in model.TIME
            )

        def _balance(model, t):
            """
            Energy balance.
            """
            sum_bids = 0.0
            for b in model.BIDS:
                try:
                    sum_bids += model.x[b] * b.flexibility[t]
                except KeyError:
                    pass

            return model.imports[t] - model.exports[t] == demand_ts_actual[t] + sum_bids - production_ts_actual[t]

        def _local_consumption(model, t):
            """
            Compute local consumption.
            """
            sum_bids = 0.0
            for b in model.BIDS:
                try:
                    sum_bids += model.x[b] * b.flexibility[t]
                except KeyError:
                    pass

            return model.local_consumption[t] == (
                demand_ts_actual[t]
                - model.imports[t]
                + sum_bids
            )

        def _local_production(model, t):
            """
            Compute local production.
            """
            return model.local_production[t] == (
                production_ts_actual[t]
                - model.exports[t]
            )

        model.Objective = Objective(rule=_objective, sense=minimize)
        model.Global_costs = Constraint(rule=_global_costs)
        model.Local_costs = Constraint(rule=_local_costs)
        model.Cost_bids = Constraint(rule=_bid_costs)
        model.Global_sales = Constraint(rule=_global_sales)
        model.Local_sales = Constraint(rule=_local_sales)
        model.Balance = Constraint(model.TIME, rule=_balance)
        model.Local_consumption = Constraint(model.TIME, rule=_local_consumption)
        model.Local_production = Constraint(model.TIME, rule=_local_production)

        model.write('test_working.lp', io_options={'symbolic_solver_labels': True})
        opt = SolverFactory(system.configuration.solver_name)
        opt.solve(model, tee=system.configuration.solver_display, keepfiles=False)

        # Bids selection
        for b in model.BIDS:
            # Retrieve bid and optimised bid acceptance
            acceptance = model.x[b].value

            # Start processing the bids: change status to pending
            b.status = BidStatus.PENDING
            # Continue processing the bid
            if b.status == BidStatus.PENDING:
                # Reserve the bid
                b.status = BidStatus.RESERVED
                # assign acceptance
                b.acceptance = acceptance
                # Filter bids according to its acceptance ratio
                if b.acceptance < EPS:
                    # Reject bid
                    b.status = BidStatus.REJECTED
                else:
                    # Accept bid
                    b.status = BidStatus.ACCEPTED

                    for t, v in b.flexibility.iteritems():
                        if t <= self._simulation_period[-1]:
                            self.initial_flexible_baseline[t] += v * b.acceptance

            else:
                # Set status as free
                b.status = BidStatus.FREE

        # Loop over the date range of interest to compute the new schedule accounting for the flexibility
        for time in date_range_of_interest_no_rebound:
            for b in flexibility_dam:
                # Filter out flexibility bids not starting at the adequate time or not accepted
                if b.flexibility.index[0] != time or b.status != BidStatus.ACCEPTED:
                    continue
                # Compute flexibility offered
                flexibility = b.acceptance * b.flexibility[time]
                # Add flexibility to the schedule of the incumbent microgrid
                b.owner.schedule[time] += flexibility

                self._compute_flexibility_costs(time, b)

    def _rt_activation(self, system, now: pd.Timestamp, time_end: pd.Timestamp):
        """
        Activates bids in real time
        """
        idle_time = 60
        payback_duration = 60
        optimization_horizon = 120

        horizon = pd.Timedelta(minutes=optimization_horizon)
        rebound = pd.Timedelta(minutes=idle_time + payback_duration)

        date_range_of_interest = pd.date_range(
            now + pd.Timedelta(minutes=15),
            now + pd.Timedelta(minutes=15) + horizon + rebound,
            freq=system.resolution_str(),
            closed='left'
        )
        # Loop over the date range of interest to retrieve flexibility bids from the real-time
        for time in date_range_of_interest:
            last_time_step = time_end - pd.Timedelta(minutes=system.resolution)
            if time <= last_time_step:
                for m in self.microgrids:
                    if m.flexibility:
                        # Create and store flexibility bids
                        self.flexibility_bids.update(m.flexibility(system, time_stamp=time, now=now, market='rt'))

        # Compute the rebound effect
        rebound_periods = max([(len(b.flexibility)) for b in self.flexibility_bids])
        date_range_of_interest_no_rebound = date_range_of_interest[:-rebound_periods]

        position = pd.Series(data=0.0, index=date_range_of_interest_no_rebound)
        forecast = pd.Series(data=0.0, index=date_range_of_interest_no_rebound)
        imbalance_ts = pd.Series(data=0.0, index=date_range_of_interest_no_rebound)
        for time in date_range_of_interest_no_rebound:
            if time <= time_end:
                for m in self.microgrids:
                    # Retrieving the position of the system
                    position[time] += m.schedule[time]
                    forecast[time] += m.baseline_forecast(time, now) + m._flex_pos[time] + m._flex_neg[time]

                position[time] += self.initial_baseline_no_microgrids[time]
                forecast[time] += self.forecaster.forecast(time, now)

            imbalance_ts[time] = position[time] - forecast[time]

        imbalance = {i+1: imbalance_ts[i] for i in range(len(imbalance_ts))}

        # Retrieve the bids starting during the date range of interest and labeled as day-ahead market
        flexibility_rt = [b for b in self.flexibility_bids
                           if (b.flexibility.index[0] in date_range_of_interest_no_rebound) and (b.market == "rt")]

        # Create multi-indexed dictionary with the bids
        bids_array = np.array([flexibility_rt[i].flexibility.values for i in range(len(flexibility_rt))])
        bids = {(i, j): bids_array[i, j] for i in range(len(bids_array)) for j in range(len(bids_array[0]))}

        # Create dictionary with bids costs
        bid_price = {i: flexibility_rt[i].cost for i in range(len(flexibility_rt))}

        # Create dictionary with dam prices
        time_improved_forecast = None  # TODO: To use for improved forecast
        prices_imbalance_pos = {t+1: system.tso.price(date_range_of_interest_no_rebound[t], time_improved_forecast)[POS]
                      for t in range(len(date_range_of_interest_no_rebound))}
        prices_imbalance_neg = {t+1: system.tso.price(date_range_of_interest_no_rebound[t], time_improved_forecast)[NEG]
                      for t in range(len(date_range_of_interest_no_rebound))}

        rebound_periods = max([(len(b.flexibility)) for b in self.flexibility_bids])

        # Create dataframe of bids
        bids_df = pd.DataFrame()
        bids_df['owner'] = [b.owner.name for b in flexibility_rt]
        bids_df['time'] = [list(date_range_of_interest_no_rebound).index(b.flexibility.index[0]) + 1
                           for b in flexibility_rt]
        bids_df['cost'] = [b.cost for b in flexibility_rt]
        bids_df['bid'] = [b for b in flexibility_rt]

        # Optimisation
        model = ConcreteModel()

        # Crete sets
        model.TIME = RangeSet(len(date_range_of_interest_no_rebound))
        model.BIDS = Set(initialize=bids_df.index.values)
        model.REBOUND = Set(initialize=range(rebound_periods))

        # Create parameters
        model.Imbalance = Param(model.TIME, initialize=imbalance)
        model.Bid = Param(model.BIDS, model.REBOUND, initialize=bids)
        model.Bid_price = Param(model.BIDS, initialize=bid_price)
        model.Imbalance_price_pos = Param(model.TIME, initialize=prices_imbalance_pos)
        model.Imbalance_price_neg = Param(model.TIME, initialize=prices_imbalance_neg)

        # Create Variables
        model.x = Var(model.BIDS, within=NonNegativeReals, bounds=(0, 1))
        model.imbalance_opt = Var(model.TIME, within=Reals)
        model.imbalance_pos = Var(model.TIME, within=NonNegativeReals)
        model.imbalance_neg = Var(model.TIME, within=NonNegativeReals)
        model.imbalance_costs = Var(within=Reals)
        model.bid_costs = Var(within=NonNegativeReals)

        def _objective(model):
            """
            Objective function to minimise total costs.
            """
            return model.imbalance_costs + model.bid_costs

        def _imbalance_costs(model):
            """
            Compute the imabalance costs.
            """

            return model.imbalance_costs >= sum(
                (
                    model.imbalance_pos[t] * model.Imbalance_price_pos[t] +
                    model.imbalance_neg[t] * model.Imbalance_price_neg[t]
                )
                for t in model.TIME
            )

        def _imbalance_splitting(model, t):
            """
            Splits the imbalance in its positive and negative components.
            """
            return model.imbalance_opt[t] == model.imbalance_pos[t] - model.imbalance_neg[t]

        def _compute_imbalance(model, t):
            """
            Optimizes the imbalance.
            """
            return model.imbalance_opt[t] == (
                model.Imbalance[t] +
                sum(model.x[b] * model.Bid[b, t-bids_df['time'][b]]
                    for b in bids_df[(bids_df['time'] <= t) & (bids_df['time'] > t-rebound_periods)].index)
            )

        def _bid_costs(model):
            """
            Compute costs of activating flexibility.
            """
            return model.bid_costs >= sum(
                model.x[b] * model.Bid_price[b]
                for b in model.BIDS
            )

        model.Objective = Objective(rule=_objective, sense=minimize)
        model.Imbalance_costs = Constraint(rule=_imbalance_costs)
        model.Imbalance_splitting = Constraint(model.TIME, rule=_imbalance_splitting)
        model.Compute_imbalance = Constraint(model.TIME, rule=_compute_imbalance)
        model.Bid_costs = Constraint(rule=_bid_costs)

        model.write('test_working.lp', io_options={'symbolic_solver_labels': True})
        opt = SolverFactory(system.configuration.solver_name)
        opt.solve(model, tee=system.configuration.solver_display, keepfiles=False)
