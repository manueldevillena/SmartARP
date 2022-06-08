import matplotlib.pyplot as plt
import os
import pandas as pd

from smartarp.core import SimulationConfiguration, System
from smartarp.utils import create_json, create_csv, read_dict_objects


def _read_data(system: System):
    """
    Read the data of the simulation and stores it in dictionaries.

    :param system: System instance of the simulation, containing all the information to be read.
    :param configuration: Configuration instance of the simulation, containing the inputs.
    :return: Three dictionaries with the relevant data.
    """
    costs = dict()
    costs_flexible = dict()
    charges = dict()
    demand = dict()

    # Costs of dam provision and imbalance (no flexibility)
    costs['cost_dam_provision'] = system.retailer.costs_dam_provision.sum()
    costs['cost_imbalance_pos'] = system.retailer.costs_imbalance_pos.sum()
    costs['cost_imbalance_neg'] = system.retailer.costs_imbalance_neg.sum()

    # Costs of dam provision and imbalance (with flexibility)
    costs_flexible['cost_dam_provision_flexible'] = system.retailer.costs_dam_provision_flexible.sum()
    costs_flexible['cost_imbalance_flex_pos'] = system.retailer.costs_imbalance_flex_pos.sum()
    costs_flexible['cost_imbalance_flex_neg'] = system.retailer.costs_imbalance_flex_neg.sum()

    # Costs of activating flexibility
    costs['cost_flexibility'] = system.retailer.costs_flexibility.sum()

    # Charges to microgrids not meeting their baselines
    charges['charge_deviation_pos'] = read_dict_objects(system.retailer.microgrid_deviation_costs_pos).values.sum()
    charges['charge_deviation_neg'] = read_dict_objects(system.retailer.microgrid_deviation_costs_neg).values.sum()

    # Demand
    demand['demand_provision'] = system.retailer.initial_baseline.sum()
    demand['demand_provision_flexible'] = system.retailer.initial_flexible_baseline.sum()
    demand['demand_realisation'] = system.retailer.demand_total.sum()
    # TODO: Re-do demands, not correct

    return costs, costs_flexible, charges, demand


def cost_analysis(system: System, configuration: SimulationConfiguration):
    """
    Creates costs analysis and outputs the results in csv and in json format.

    :param system: System instance of the simulation, containing all the information to be read.
    :param configuration: Configuration instance of the simulation, containing the inputs.
    """
    if configuration.use_dam_flexibility and configuration.use_rt_flexibility:
        flexibility = 'all_flex'
    elif configuration.use_dam_flexibility:
        flexibility = 'dam_flex'
    elif configuration.use_rt_flexibility:
        flexibility = 'rt_flex'
    else:
        flexibility = 'no_flex'

    os.makedirs(configuration.path_output, exist_ok=True)

    costs, costs_flexible, charges, demand = _read_data(system=system)

    costs_minus_charges = sum(costs.values()) - sum(charges.values())
    costs_minus_charges_flexible = sum(costs_flexible.values()) - sum(charges.values())

    cost_per_mwh_provision = costs_minus_charges / demand['demand_provision']
    if demand['demand_provision_flexible'] > 0:
        cost_per_mwh_provision_flexible = costs_minus_charges_flexible / demand['demand_provision_flexible']
    else:
        cost_per_mwh_provision_flexible = 0
    cost_per_mwh_realisation = costs_minus_charges / demand['demand_realisation']
    cost_per_mwh_realisation_flexible = costs_minus_charges_flexible / demand['demand_realisation']

    cost_analysis_results = dict(costs_dam_prov=costs['cost_dam_provision'],
                                 costs_imbalace_pos=costs['cost_imbalance_pos'],
                                 costs_imbalace_neg=costs['cost_imbalance_neg'],
                                 cost_flex=costs['cost_flexibility'],
                                 costs_total=sum(costs.values()),
                                 charges=sum(charges.values()),
                                 costs_minus_charges=costs_minus_charges,
                                 demand_provision=demand['demand_provision'],
                                 demand_provision_flexible=demand['demand_provision_flexible'],
                                 demand_realisation=demand['demand_realisation'],
                                 cost_per_mwh_provision=cost_per_mwh_provision,
                                 cost_per_mwh_provision_flexible=cost_per_mwh_provision_flexible,
                                 cost_per_mwh_realisation=cost_per_mwh_realisation,
                                 cost_per_mwh_realisation_flexible=cost_per_mwh_realisation_flexible,
                                 bids_accepted=sum(m.bids_number for m in system.retailer.microgrids)
                                 )

    create_csv(cost_analysis_results, configuration.path_output, 'cost_analysis_results_{}'.format(flexibility),
               direct=True)
    create_json(cost_analysis_results, configuration.path_output, 'cost_analysis_results_{}'.format(flexibility),
                direct=True)
    _output_plot_cost_analysis(system, configuration, name=flexibility)


def _output_plot_cost_analysis(system: System, configuration: SimulationConfiguration, name: str):
    """
    Plots the results of the simulations (only cost analysis).

    :param system: System instance of the simulation, containing all the information to be read.
    :param configuration: Configuration instance of the simulation, containing the inputs.

    """
    os.makedirs(configuration.path_output, exist_ok=True)

    costs, costs_flexible, charges, demand = _read_data(system=system)

    costs_minus_charges = sum(costs.values()) - sum(charges.values())
    costs_minus_charges_flexible = sum(costs_flexible.values()) - sum(charges.values())
    df_to_plot = pd.DataFrame(index=[name])

    df_to_plot['cost_per_mwh_provision'] = costs_minus_charges / demand['demand_provision']
    df_to_plot['cost_per_mwh_realisation'] = costs_minus_charges / demand['demand_realisation']
    if system.retailer.use_dam_flexibility:
        df_to_plot['cost_per_mwh_provision_flexible'] = costs_minus_charges_flexible / demand['demand_provision_flexible']
        df_to_plot['cost_per_mwh_realisation_flexible'] = costs_minus_charges_flexible / demand['demand_realisation']
    else:
        df_to_plot['cost_per_mwh_provision_flexible'] = 0.0
        df_to_plot['cost_per_mwh_realisation_flexible'] = 0.0

    fig, ax = plt.subplots(figsize=(15, 10))
    df_to_plot.plot(ax=ax, kind='bar')

    plt.ylabel('EUR/MWh')
    plt.xticks([])
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig(os.path.join(configuration.path_output, 'cost_analysis.pdf'))
    plt.close(fig)
