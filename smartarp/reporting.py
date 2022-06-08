import json
import matplotlib.pyplot as plt
import os
import pandas as pd


def read_results(dict_simulations: dict, list_values_to_report_1: list, list_values_to_report_2: list,
                 list_values_to_report_3: list):
    """

    :param dict_simulations:
    :param list_values_to_report:
    :return:
    """
    df_to_plot_per_mwh = pd.DataFrame(index=list_values_to_report_1)
    df_to_plot_absolute = pd.DataFrame(index=list_values_to_report_2)
    df_to_plot_demand = pd.DataFrame(index=list_values_to_report_3)
    for key, value in dict_simulations.items():
        with open(value, "r") as read_file:
            data = json.load(read_file)

        df_to_plot_per_mwh[key] = [
            data['cost_per_mwh_provision'],
            data['cost_per_mwh_realisation'],
            data['cost_per_mwh_provision_flexible']
        ]
        df_to_plot_absolute[key] = [
            data['cost_flex'] / 1000,
            data['charges'] / 1000
        ]
        df_to_plot_demand[key] = [
            data['demand_provision'] / 1000,
            data['demand_realisation'] / 1000,
            data['demand_provision_flexible'] / 1000
        ]
    return df_to_plot_per_mwh, df_to_plot_absolute, df_to_plot_demand


def output_plot_cost_analysis(data: pd.DataFrame, ylabel: str, path_output: str):
    """
    Plots the results of the simulations (only cost analysis).

    :param data: Dataframe with the information to plot.
    :param ylabel: Label of the Y axis.
    :param path_output: Path to store the plot in pdf.
    """
    fig, ax = plt.subplots()
    data.plot(ax=ax, kind='bar', rot=5, grid=True)

    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend(loc=3)
    plt.savefig(path_output)
    plt.close(fig)


if __name__ == "__main__":
    dict_simulations = {
        'no_flex': '/Users/villena/bitbucket/smartarp_simulator/results/no_flex_test/cost_analysis_results_no_flex.json',
        'all_flex': '/Users/villena/bitbucket/smartarp_simulator/results/dam_flex_test/cost_analysis_results_dam_flex.json'
    }
    list_values_to_report_per_mwh = [
        'cost_per_mwh_provision',
        'cost_per_mwh_realisation',
        'cost_per_mwh_provision_flexible',
    ]
    list_values_to_report_absolute = [
        'cost_flex',
        'charges'
    ]
    list_values_to_report_demand = [
        'demand_provision',
        'demand_realisation',
        'demand_provision_flexible',
    ]

    dfs_to_plot = read_results(dict_simulations,
                               list_values_to_report_per_mwh,
                               list_values_to_report_absolute,
                               list_values_to_report_demand)
    output_plot_cost_analysis(dfs_to_plot[0], ylabel='EUR/MWh', path_output='cost_analysis_per_mwh_price.pdf')
    output_plot_cost_analysis(dfs_to_plot[1], ylabel='kEUR', path_output='cost_analysis_per_mwh_charge.pdf')
    output_plot_cost_analysis(dfs_to_plot[2], ylabel='GWh', path_output='cost_analysis_absolute.pdf')
