import json
import pandas as pd


class SimulationConfiguration:
    """
    Sets the agents' configuration.
    """
    def __init__(self, path_input: str = '', path_output: str = ''):

        self.path_input = path_input
        self.path_output = path_output
        self.time_start = None
        self.time_end = None
        self.resolution = 15
        self.path_dam_prices = None
        self.path_imbalance_prices = None
        self.path_demand_retailer = None
        self.demand_conversion_to_MWh = 250  #: conversion factor to adapt the units of the 15 minutes resolution to MWh.
        self.path_demand_amr = None
        self.microgrids = dict()
        self.retail_price = 60.0
        self.deviation_ratio = 0.1
        self.cost_per_bid = 0.1
        self.use_dam_flexibility = True
        self.use_rt_flexibility = False

        # Economic analysis
        self.purchasing_price_global = 60.0
        self.purchasing_price_local = 56.0
        self.selling_price_global = 40.0
        self.selling_price_local = 55.0
        self.distribution_price_global = 85.0
        self.distribution_price_local = 0.79
        self.taxes_price = 75.0
        self.retailer_mode = False

        # Solver
        self.solver_name = 'cplex'
        self.solver_display = False

    def load(self, parsed_arguments=None):
        """
        Loads the parameters of the simulation from the json file.

        :param parsed_arguments: Dictionary of parsed arguments.
        """
        with open(self.path_input, 'r') as reader:
            data = reader.read()
        inputs = json.loads(data)

        # Mandatory attributes
        for attr in [
            'time_start', 'time_end', 'path_dam_prices', 'path_imbalance_prices', 'path_demand_retailer',
            'path_demand_amr', 'microgrids'
        ]:
            try:
                setattr(self, attr, inputs[attr])
            except KeyError:
                raise KeyError('Attribute "{}" is mandatory in the configuration file.'.format(attr))

        # Optional attributes
        for attr in [
            'resolution', 'retail_price', 'deviation_ratio', 'cost_per_bid', 'use_dam_flexibility',
            'use_rt_flexibility', 'purchasing_price_global', 'purchasing_price_local', 'selling_price_global',
            'selling_price_local', 'distribution_price_global', 'distribution_price_local', 'taxes_price',
            'retailer_mode'
        ]:
            try:
                setattr(self, attr, inputs[attr])
            except KeyError:
                pass

        # Dependant parameters
        self.demand_conversion_to_MWh = 1000 * self.resolution / 60

        # Solver
        if parsed_arguments is not None:
            self.solver_name = parsed_arguments.solver

    def resolution_str(self):
        """
        Get the resolution pandas style.

        :return: String.
        """
        return '{}T'.format(self.resolution)

    def simulation_period(self):
        """
        Creates the simulation horizon.

        :return:
        """
        return pd.date_range(start=self.time_start, end=self.time_end, freq=self.resolution_str(), tz='UTC')

    def simulation_period_hourly(self):
        """
        Creates the simulation horizon.

        :return:
        """
        return pd.date_range(start=self.time_start, end=self.time_end, freq='H', tz='UTC')

    def to_json(self, file_name='configuration.json'):
        """
        Save itself to a json file.
        """
        d = {a: getattr(self, a) for a in self.__dict__ if not a.startswith('_')}
        with open('{}/{}'.format(self.path_output, file_name), 'w') as f:
            json.dump(d, f, indent=4)
