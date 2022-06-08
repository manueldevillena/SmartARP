import matplotlib.pyplot as plt
import os
import pandas as pd

from copy import deepcopy
from pandas.plotting import register_matplotlib_converters
from smartarp.core import SimulationConfiguration, System
from smartarp.utils import create_csv, read_dict_objects, create_colours, create_json

register_matplotlib_converters()

POS = 'pos'
NEG = 'neg'


def _read_data(system: System, configuration: SimulationConfiguration):
    """
    Read the data of the simulation and stores it in dictionaries.

    :param system: System instance of the simulation, containing all the information to be read.
    :param configuration: Configuration instance of the simulation, containing the inputs.
    :return: Two dictionaries. The first one with all the data. The second one with the relevant figures.
    """
    time_simulation = [configuration.simulation_period()[0], configuration.simulation_period()[-1]]
    metrics, demand_shift = _compute_metrics(configuration, system)

    data = dict()
    # energy prices
    data['dam_prices'] = system.dam.prices[time_simulation[0]:time_simulation[-1]]
    # retailer's demand without microgrids
    data['demand_base'] = system.retailer.demand_base[time_simulation[0]:time_simulation[-1]]
    # retailer's demand with microgrids
    data['demand_total'] = system.retailer.demand_total
    # retailer's revenues with microgrids
    data['revenues'] = system.retailer.revenues
    # retailer's predicted demand with microgrids
    data['initial_baseline'] = system.retailer.initial_baseline
    # retailer's predicted demand with microgrids and flexibility activated in dam
    data['initial_flexible_baseline'] = system.retailer.initial_flexible_baseline
    # flexibility upward in the day-ahead
    data['dam_flexibility_upward'] = system.retailer.dam_flexibility_upward
    # flexibility downward in the day-ahead
    data['dam_flexibility_downward'] = system.retailer.dam_flexibility_downward
    # microgrids' baselines
    data['baseline_microgrids'] = read_dict_objects(system.retailer.microgrid_baselines)
    # microgrids' retailing costs
    data['baseline_retailing_costs'] = read_dict_objects(system.retailer.microgrid_retailing_costs)
    # microgrids' positive deviation costs
    data['baseline_deviation_costs_pos'] = read_dict_objects(system.retailer.microgrid_deviation_costs_pos)
    # microgrids' negative deviation costs
    data['baseline_deviation_costs_neg'] = read_dict_objects(system.retailer.microgrid_deviation_costs_neg)
    # positive imbalance prices
    data['imbalance_prices_pos'] = system.tso.imbalance_prices[time_simulation[0]:time_simulation[-1]][POS]
    # negative imbalance prices
    data['imbalance_prices_neg'] = system.tso.imbalance_prices[time_simulation[0]:time_simulation[-1]][NEG]
    # positive imbalance
    data['energy_imbalance_pos'] = system.retailer.energy_imbalance_pos
    # negative imbalance
    data['energy_imbalance_neg'] = system.retailer.energy_imbalance_neg
    # costs for day-ahead market provision
    data['costs_dam_provision'] = system.retailer.costs_dam_provision_flexible
    # costs for positive imbalances
    data['costs_imbalance_pos'] = system.retailer.costs_imbalance_flex_pos
    # costs for negative imbalances
    data['costs_imbalance_neg'] = system.retailer.costs_imbalance_flex_neg
    # costs for positive flexibility
    data['costs_flexibility'] = system.retailer.costs_flexibility
    # costs for distribution global
    data['costs_distribution_global'] = system.retailer.initial_flexible_baseline * \
                                         (1 - metrics['self_sufficiency_rate_flexible'])
    # costs for distribution local
    data['costs_distribution_local'] = system.retailer.initial_flexible_baseline * \
                                         metrics['self_sufficiency_rate_flexible']
    # costs for taxes
    data['costs_taxes'] = system.retailer.initial_flexible_baseline * configuration.taxes_price
    return data


def _compute_costs(system, configuration):
    """

    :param system:
    :param configuration:
    :return:
    """
    time_simulation = configuration.simulation_period()
    metrics, demand = _compute_metrics(configuration, system)

    # COSTS NO REC
    cash_flow_no_rec = pd.DataFrame(index=time_simulation)
    cash_flow_no_rec['energy'] = demand['demand'] * configuration.purchasing_price_global
    cash_flow_no_rec['distribution_global'] = demand['demand'] * configuration.distribution_price_global
    cash_flow_no_rec['taxes'] = demand['demand'] * configuration.taxes_price
    cash_flow_no_rec['production_local'] = pd.Series(data=0.0, index=time_simulation)
    cash_flow_no_rec['production_global'] = -demand['production'] * configuration.selling_price_global

    # COSTS REC NO FLEXIBILITY
    cash_flow_rec_no_flex = pd.DataFrame(index=time_simulation)
    cash_flow_rec_no_flex['energy'] = (
            demand['demand'] * (1 - metrics['self_sufficiency_rate_initial']) * configuration.purchasing_price_global +
            demand['demand'] * metrics['self_sufficiency_rate_initial'] * configuration.purchasing_price_local
    )
    cash_flow_rec_no_flex['distribution_global'] = (
            demand['demand'] * (1 - metrics['self_sufficiency_rate_initial']) * configuration.distribution_price_global
    )
    cash_flow_rec_no_flex['distribution_local'] = (
            demand['demand'] * metrics['self_sufficiency_rate_initial'] *
            configuration.distribution_price_global * configuration.distribution_price_local
    )
    cash_flow_rec_no_flex['taxes'] = demand['demand'] * configuration.taxes_price
    cash_flow_rec_no_flex['production_local'] = -demand['used_production_initial'] * configuration.selling_price_local
    cash_flow_rec_no_flex['production_global'] = -demand['spilled_production_initial'] * configuration.selling_price_global

    # COSTS REC FLEXIBILITY
    cash_flow_rec_flex = pd.DataFrame(index=time_simulation)
    cash_flow_rec_flex['energy'] = (
            demand['demand_flexible'] * (1 - metrics['self_sufficiency_rate_flexible']) * configuration.purchasing_price_global +
            demand['demand_flexible'] * metrics['self_sufficiency_rate_flexible'] * configuration.purchasing_price_local
    )
    cash_flow_rec_flex['distribution_global'] = (
            demand['demand_flexible'] * (1 - metrics['self_sufficiency_rate_flexible']) * configuration.distribution_price_global
    )
    cash_flow_rec_flex['distribution_local'] = (
            demand['demand_flexible'] * metrics['self_sufficiency_rate_flexible'] *
            configuration.distribution_price_global * configuration.distribution_price_local
    )
    cash_flow_rec_flex['taxes'] = demand['demand_flexible'] * configuration.taxes_price
    cash_flow_rec_flex['production_local'] = -demand['used_production_flexible'] * configuration.selling_price_local
    cash_flow_rec_flex['production_global'] = -demand['spilled_production_flexible'] * configuration.selling_price_global

    total_cash_flow = {
        'cash_flow_no_rec': cash_flow_no_rec.sum(axis=1).sum(axis=0),
        'cash_flow_rec_no_flex': cash_flow_rec_no_flex.sum(axis=1).sum(axis=0),
        'cash_flow_rec_flex': cash_flow_rec_flex.sum(axis=1).sum(axis=0)
    }

    return cash_flow_no_rec, cash_flow_rec_no_flex, cash_flow_rec_flex, total_cash_flow


def _retrieve_demand_forecast(system: System, configuration: SimulationConfiguration):
    """
    Retrieves the data from the simulation.

    :param system: System instance of the simulation, containing all the information to be read.
    :param configuration: Configuration instance of the simulation, containing the inputs.
    :return: Three dictionaries with the consumption of the consumers, the consumption of the microgrids, and the
    production of the microgrids.
    """
    time_simulation = configuration.simulation_period()

    energy_consumed = {
        'Consumers': system.retailer.initial_baseline_no_microgrids[time_simulation[0]:time_simulation[-1]],
        'Flexible consumers': pd.Series(data=0.0, index=time_simulation)
    }
    consumption_customers = {
        'Non-flexible consumers': system.retailer.initial_baseline_no_microgrids[time_simulation[0]:time_simulation[-1]]
    }
    consumption_microgrids = {}
    production = {}
    total_production = pd.Series(data=0.0, index=time_simulation)
    for m in system.retailer.microgrids:
        series = system.retailer.microgrid_baselines[m][time_simulation[0]:time_simulation[-1]]
        if series[series > 0].any():
            energy_consumed['Flexible consumers'] += series
            consumption_microgrids[m.name] = series
        else:
            production[m.name] = -series
            total_production += -series

    net_energy = {'net_energy': energy_consumed['Consumers'] + energy_consumed['Flexible consumers'] - total_production}

    return consumption_customers, consumption_microgrids, production, energy_consumed, net_energy


def _retrieve_demand_realisation(system: System, configuration: SimulationConfiguration):
    """
    Retrieves the data from the simulation.
from smartarp.utils import create_json, create_csv

    :param system: System instance of the simulation, containing all the information to be read.
    :param configuration: Configuration instance of the simulation, containing the inputs.
    :return: Three dictionaries with the consumption of the consumers, the consumption of the microgrids, and the
    production of the microgrids.
    """
    time_simulation = configuration.simulation_period()

    energy_consumed = {
        'Consumers': system.retailer.demand_base[time_simulation[0]:time_simulation[-1]],
        'Flexible consumers': pd.Series(data=0.0, index=time_simulation)
    }
    consumption_customers = {
        'Non-flexible consumers': system.retailer.demand_base[time_simulation[0]:time_simulation[-1]]
    }
    consumption_microgrids = {}
    production = {}
    total_production = pd.Series(data=0.0, index=time_simulation)
    for m in system.retailer.microgrids:
        series = system.retailer.microgrid_consumption[m][time_simulation[0]:time_simulation[-1]]
        if series[series > 0].any():
            energy_consumed['Flexible consumers'] += series
            consumption_microgrids[m.name] = series
        else:
            production[m.name] = -series
            total_production += -series

    net_energy = {'net_energy': energy_consumed['Consumers'] + energy_consumed['Flexible consumers'] - total_production}

    return consumption_customers, consumption_microgrids, production, energy_consumed, net_energy


def _retrieve_flexibility(system: System):
    """
    Retrieves the flexibility activated by the microgrids.

    :return:
    """
    positive_flexibility_activated = pd.DataFrame()
    negative_flexibility_activated = pd.DataFrame()
    for m in system.retailer.microgrids:
        positive_flexibility_activated[m.name] = m.flex_pos
        negative_flexibility_activated[m.name] = m.flex_neg

    return positive_flexibility_activated, negative_flexibility_activated


def _compute_rate(numerator: pd.Series, denominator: pd.Series) -> float:
    """
    Computes the self-sufficiency rate of the REC.
    """
    if numerator is pd.Series and denominator is pd.Series:
        return numerator.divide(denominator, fill_value=0)
    else:
        return float(numerator / denominator)


def _compute_metrics(configuration, system):
    """
    Computes all the metrics to report.
    """
    time_simulation = configuration.simulation_period()

    demand_base_retailer = system.retailer.demand_base

    demand = pd.Series(data=0.0, index=time_simulation)
    production = pd.Series(data=0.0, index=time_simulation)
    for t in time_simulation:
        for m in system.retailer.microgrids:
            if not m.producer:
                demand[t] += m.initial_demand[t]
            else:
                production[t] += abs(m.initial_demand[t])
        demand[t] += demand_base_retailer[t]

    demand_flexible = pd.Series(data=0.0, index=time_simulation)
    production_flexible = pd.Series(data=0.0, index=time_simulation)
    for t in time_simulation:
        for m in system.retailer.microgrids:
            if not m.producer:
                demand_flexible[t] += m.consumption(t)
            else:
                production_flexible[t] += abs(m.consumption(t))
        demand_flexible[t] += demand_base_retailer[t]

    spilled_production = pd.Series(data=0.0, index=time_simulation)
    spilled_production_flexible = pd.Series(data=0.0, index=time_simulation)
    used_production = pd.Series(data=0.0, index=time_simulation)
    used_production_flexible = pd.Series(data=0.0, index=time_simulation)
    for t in time_simulation:
        spilled_production[t] += abs(min(0.0, demand[t] - production[t]))
        spilled_production_flexible[t] += abs(min(0.0, demand_flexible[t] - production_flexible[t]))
        used_production[t] += production[t] - spilled_production[t]
        used_production_flexible[t] += production_flexible[t] - spilled_production_flexible[t]

    metrics = dict()
    # Compute metrics before flexibility
    metrics['self_sufficiency_rate_initial'] = _compute_rate(used_production.sum(), demand.sum())
    metrics['self_consumption_rate_initial'] = _compute_rate(used_production.sum(), production.sum())
    metrics['coverage_rate_initial'] = _compute_rate(production.sum(), demand.sum())
    metrics['spilled_production_initial'] = spilled_production.sum()
    metrics['demand'] = demand.sum()
    metrics['used_production_initial'] = used_production.sum()

    # Compute metrics after flexibility
    metrics['self_sufficiency_rate_flexible'] = _compute_rate(used_production_flexible.sum(), demand_flexible.sum())
    metrics['self_consumption_rate_flexible'] = _compute_rate(used_production_flexible.sum(), production_flexible.sum())
    metrics['coverage_rate_flexible'] = _compute_rate(production_flexible.sum(), demand_flexible.sum())
    metrics['spilled_production_flexible'] = spilled_production_flexible.sum()
    metrics['demand_flexible'] = demand_flexible.sum()
    metrics['production'] = production.sum()
    metrics['used_production_flexible'] = used_production_flexible.sum()

    demand_shift = pd.DataFrame(index=time_simulation)
    demand_shift['demand'] = demand
    demand_shift['demand_flexible'] = demand_flexible
    demand_shift['production'] = production
    demand_shift['used_production_initial'] = used_production
    demand_shift['used_production_flexible'] = used_production_flexible
    demand_shift['spilled_production_initial'] = spilled_production
    demand_shift['spilled_production_flexible'] = spilled_production_flexible

    return metrics, demand_shift


def _computation_costs_microgrids(configuration, system, metrics):
    """

    :param configuration:
    :param system:
    """
    time_simulation = configuration.simulation_period()
    microgrid_retailing_costs = dict()
    for m in system.retailer.microgrids:
        microgrid_retailing_costs[m] = pd.Series(data=0.0, index=time_simulation)

    for t in time_simulation:
        for m in system.retailer.microgrids:
            microgrid_retailing_costs[m][t] = (
                    (
                            (m.consumption(t) * (1 - metrics['self_sufficiency_rate_flexible'])) *
                            (
                                    configuration.purchasing_price_global +
                                    configuration.distribution_price_global +
                                    configuration.taxes_price
                            )
                    )
                    +
                    (
                            (m.consumption(t) * metrics['self_sufficiency_rate_flexible']) *
                            (
                                    configuration.purchasing_price_local +
                                    configuration.distribution_price_local +
                                    configuration.taxes_price
                            )
                     )
            )

    return microgrid_retailing_costs


def _plot_demand_shift(demand_shift: pd.DataFrame, output_path):
    """
    Plots the demand shift

    """
    list_discard = ['used_production_initial', 'used_production_flexible', 'spilled_production_initial',
                    'spilled_production_flexible']
    kwds = {'color': ['gray', 'black', 'orange']}
    df_to_plot = pd.DataFrame()
    for col in demand_shift.columns:
        if col not in list_discard:
            df_to_plot[col] = demand_shift[col]
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df_to_plot['demand'], lw=1, color='red', alpha=0.7, label='demand')
    plt.plot(df_to_plot['demand_flexible'], lw=1, color='blue', alpha=0.7, label='flexible demand')
    plt.fill_between(df_to_plot.index, df_to_plot['production'], color='orange', alpha=0.5, lw=0,
                     label='production')
    plt.grid()
    plt.legend(fontsize=15)
    plt.savefig(output_path)
    plt.close(fig)


def _plot_timeseries(ts: pd.Series, y_label, output_path: str):
    """
    Plot a pandas time series.

    :param ts: Time series.
    :param output_path: Complete output file_path
    :param y_label: Y label.
    """
    fig, ax = plt.subplots(figsize=(15, 10))

    ts.plot(ax=ax)

    plt.xlabel("Time")
    if y_label is not None:
        plt.ylabel(y_label)
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig(output_path)
    plt.close(fig)


def _plot_energy(df: pd.DataFrame, y_label, output_path: str):
    """
    Plot energy_provision retailer.

    :param df: Dataframe with the energy timeseries to plot.
    :param y_label: Y label.
    :param output_path: Path to store the produced pdf.
    """

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(df.sum(axis=1), lw=1.5, ls='-', color='red', alpha=.8, label='Net energy')
    df.plot(ax=ax, kind='area', stacked=True, alpha=.4)

    ax.legend(fontsize=15)
    ax.set_xlabel("Time")
    if y_label is not None:
        ax.set_ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def _plot_imbalance(df: pd.DataFrame, y_label, output_path: str):
    """
    Plot imbalance prices.

    :param df: Dataframe with the imbalance prices to plot.
    :param y_label: Y label.
    :param output_path: Path to store the produced pdf.
    """
    fig, ax = plt.subplots(figsize=(15, 10))

    df.plot(ax=ax)

    plt.xlabel("Time")
    if y_label is not None:
        plt.ylabel(y_label)
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig(output_path)
    plt.close(fig)


def _plot_bins_flexibility(df: pd.DataFrame, df2: pd.DataFrame, y_label, output_path: str):
    """
    Plot costs retailer.

    :param df: Dataframe with the costs timeseries to plot.
    :param y_label: Y label.
    :param output_path: Path to store the produced pdf.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    df.plot(kind='bar', stacked=True, ax=axes[0], legend=False)
    df2.plot(kind='bar', stacked=True, ax=axes[1], legend=False)

    axes[0].tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    axes[1].tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    axes[1].set_xlabel("Time")
    if y_label is not None:
        axes[0].set_ylabel(y_label)
        axes[1].set_ylabel(y_label)
    fig.tight_layout()
    # axes[0].legend(fontsize=15)
    # axes[1].legend(fontsize=15)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_bins(df: pd.DataFrame, y_label, output_path: str, line_df: pd.DataFrame = None):
    """
    Plot costs retailer.

    :param df: Dataframe with the costs timeseries to plot.
    :param y_label: Y label.
    :param output_path: Path to store the produced pdf.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    df.plot(ax=ax, kind='bar', stacked=True)
    if line_df is not None:
        line_df.plot(ax=ax)

    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    plt.xlabel("Time")
    if y_label is not None:
        plt.ylabel(y_label)
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig(output_path)
    plt.close(fig)


def _plot_costs_microgrids(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, y_label, output_path: str):
    """
    Plot costs retailer.

    :param df: Dataframe with the costs timeseries to plot.
    :param y_label: Y label.
    :param output_path: Path to store the produced pdf.
    """
    df = pd.concat(
        [
            df1.to_frame(name='Retailing costs'),
            df2.to_frame(name='Positive deviation costs'),
            df3.to_frame(name='Negative deviation costs')
        ],
        axis=1
    )

    fig, ax = plt.subplots(figsize=(15, 10))
    df.plot(ax=ax, kind='bar', stacked=True)

    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    plt.xlabel("Time")
    if y_label is not None:
        plt.ylabel(y_label)
    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.savefig(output_path)
    plt.close(fig)


def _plot_dict_ts(dict_ts, axes, legend: bool = True):
    """
    Plots the timeseries stored in a dictionary with the correct labels.

    :param dict_ts: Dictionary containing the name and the timeseries to plot.
    :param axes: Axis of the plot.
    """
    for name, ts in dict_ts.items():
        axes.step(ts.index, ts, label=name)
        axes.set_xticks([])
    if legend:
        axes.legend()
    axes.grid()


def _plot_pv_output(consumption_consumers: dict, consumption_microgrids: dict, production: dict, net_energy: dict,
                    output_path):
    """
    Plots the consumption and pv output of consumers and microgrid.

    :param consumption_consumers: Timeseries with data of consumer's consumption.
    :param consumption_microgrids: Dictionary with consumption data.
    :param production: Dictionary with production data.
    :param output_path: Path to store the plots.
    """
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10, 7))
    _plot_dict_ts(consumption_consumers, ax[0])
    _plot_dict_ts(consumption_microgrids, ax[1], legend=False)
    _plot_dict_ts(production, ax[2])
    _plot_dict_ts(net_energy, ax[3])
    plt.savefig(output_path)


def _plot_production_consumption_stacked(energy: dict, production: dict, ylabel: str, output_path: str):
    """
    Plots the consumption and the production in one stacked plot.

    """
    colours = create_colours(len(energy))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    y0 = 0
    y1 = 0
    colour = 0
    for key, series in energy.items():
        y1 += series.values
        ax.fill_between(range(len(series)), y0, y1, step='post', color=colours[colour], alpha=0.7, label=key)
        y0 = deepcopy(y1)
        colour += 1

    production_series = 0
    for key, series in production.items():
        production_series += series.values
    ax.fill_between(range(len(production_series)), 0, production_series, step='post', color='orange', label='Production')
    ax.step(range(len(y1)), y1 - production_series, color='red', label='Net Energy')

    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    plt.legend()
    plt.savefig(output_path)


def _create_dataframe_consumption(system: System):
    """
    Reads the data needed to compute the repartition of keys.
    """
    time_simulation = system.configuration.simulation_period()

    all_consumers = system.retailer.individual_demands.columns
    flexible_consumers = list()

    consumption = pd.DataFrame()
    for m in system.retailer.microgrids:
        name = m.name
        flexible_consumers.append(name)
        data_to_store = pd.Series(data=0.0, index=time_simulation)
        for t in time_simulation:
            data_to_store[t] = m.consumption(t)
        consumption[name] = data_to_store

    non_flexible_consumers = [x for x in all_consumers if x not in flexible_consumers]
    for consumer in non_flexible_consumers:
        consumption[consumer] = system.retailer.individual_demands[consumer]

    return consumption


def output_create_results(system: System, configuration: SimulationConfiguration):
    """
    Creates output files (csv and json).

    :param system: System instance of the simulation, containing all the information to be read.
    :param configuration: Configuration instance of the simulation, containing the inputs.
    """
    os.makedirs(configuration.path_output, exist_ok=True)

    configuration.to_json()
    data = _read_data(system, configuration)
    metrics, demand_shift = _compute_metrics(configuration, system)
    cash_flow_no_rec, cash_flow_rec_no_flex, cash_flow_rec_flex, total_cash_flow = _compute_costs(system, configuration)
    df_consumption = _create_dataframe_consumption(system)

    create_csv(data, configuration.path_output)
    create_csv(metrics, configuration.path_output, 'metrics', direct=True)
    create_json(metrics, configuration.path_output, 'metrics', direct=True)
    create_csv(total_cash_flow, configuration.path_output, 'costs_total', direct=True)
    create_json(total_cash_flow, configuration.path_output, 'costs_total', direct=True)
    create_csv(df_consumption, configuration.path_output, 'consumption_optimized', from_df=True)
    create_csv(demand_shift, configuration.path_output, 'timeseries_optimization', from_df=True)


def output_plot_timeseries(system: System, configuration: SimulationConfiguration):
    """
    Plots the results of the simulation (only time series).

    :param system: System instance of the simulation, containing all the information to be read.
    :param configuration: Configuration instance of the simulation, containing the inputs.
    """
    os.makedirs(configuration.path_output, exist_ok=True)

    data = _read_data(system, configuration)
    consumption_customers_fore, consumption_microgrids_fore, production_fore, energy_consumed_fore, net_energy_fore = \
        _retrieve_demand_forecast(system, configuration)
    consumption_customers_real, consumption_microgrids_real, production_real, energy_consumed_real, net_energy_real = \
        _retrieve_demand_realisation(system, configuration)
    metrics, demand_shift = _compute_metrics(configuration, system)
    cash_flow_no_rec, cash_flow_rec_no_flex, cash_flow_rec_flex, total_cash_flow = _compute_costs(system, configuration)

    _plot_bins(cash_flow_no_rec, "Costs [€]",
               os.path.join(configuration.path_output, "cash_flow_no_rec.pdf"))
    _plot_bins(cash_flow_rec_no_flex, "Costs [€]",
               os.path.join(configuration.path_output, "cash_flow_rec_no_flex.pdf"))
    _plot_bins(cash_flow_rec_flex, "Costs [€]",
               os.path.join(configuration.path_output, "cash_flow_rec_flex.pdf"))

    # generation of the dataframe with the predicted demand
    _plot_production_consumption_stacked(energy_consumed_fore, production_fore, "Energy [MWh]",
                                         os.path.join(configuration.path_output, "energy_prediction_stacked.pdf"))

    # generation of the dataframe with the realisation of the demand
    _plot_production_consumption_stacked(energy_consumed_real, production_real, "Energy [MWh]",
                                         os.path.join(configuration.path_output, "energy_realisation_stacked.pdf"))

    flexibility = pd.concat(
        [
            data['initial_baseline'].to_frame(name='Initial baseline'),
            data['initial_flexible_baseline'].to_frame(name='Initial flexible baseline')
        ],
        axis=1
    )
    _plot_timeseries(flexibility, "Demand [kWh]",
                     os.path.join(configuration.path_output, "flexibility_dam.pdf"))

    # flexibility
    positive_flexibility, negative_flexibility = _retrieve_flexibility(system)
    _plot_bins_flexibility(positive_flexibility, negative_flexibility, "Flexibility [MWh]",
                           os.path.join(configuration.path_output, "flexibility_microgrids.pdf"))

    # imbalance prices
    imbalance_prices = pd.concat(
        [
            data['imbalance_prices_pos'].to_frame(name='Positive imbalance prices'),
            data['imbalance_prices_neg'].to_frame(name='Negative imbalance prices')
        ],
        axis=1
    )
    _plot_imbalance(
        imbalance_prices, "Imbalance Prices [€/MWh]",
        os.path.join(configuration.path_output, "imbalance_prices.pdf")
    )

    # DAM prices
    dam_prices = data['dam_prices']
    _plot_timeseries(
        dam_prices, "DAM prices", os.path.join(configuration.path_output, "dam_prices.pdf")
    )

    # microgrid costs
    microgrid_retailing_costs = read_dict_objects(_computation_costs_microgrids(configuration, system, metrics))
    microgrid_deviation_costs_pos = data['baseline_deviation_costs_pos']
    microgrid_deviation_costs_neg = data['baseline_deviation_costs_neg']
    microgrids = microgrid_retailing_costs.columns
    for m in microgrids:
        _plot_costs_microgrids(
            microgrid_retailing_costs[m], microgrid_deviation_costs_pos[m], microgrid_deviation_costs_neg[m],
            "Costs [€]", os.path.join(configuration.path_output, "{}_costs.pdf".format(m))
        )

    # Consumption and production
    _plot_pv_output(
        consumption_customers_fore, consumption_microgrids_fore, production_fore, net_energy_fore,
        os.path.join(configuration.path_output, "consumption_production_fore.pdf")
    )
    _plot_pv_output(
        consumption_customers_real, consumption_microgrids_real, production_real, net_energy_real,
        os.path.join(configuration.path_output, "consumption_production_real.pdf")
    )

    _plot_demand_shift(
        demand_shift, os.path.join(configuration.path_output, "demand_shift.pdf"))
