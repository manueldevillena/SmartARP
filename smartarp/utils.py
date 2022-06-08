import json
import os

import numpy as np
import pandas as pd

from matplotlib import cm

# def create_json(path_data, file):
#     """
#     Creates json file.
#
#     :param path_data: Path to all data files.
#     :param required_data: Desired csv.
#     :return: Local json file.
#     """
#     dam_prices_path = os.path.join(path_data, file)
#     dict_data = dict(dam_prices_path=dam_prices_path)
#     json_string = json.dumps(dict_data)
#
#     return json_string


# def save_json(data):
#     """
#     Stores the json file in disk.
#
#     """
#     with open('data.json', 'w') as outfile:
#         json.dump(data, outfile)


def unfold_json(data):
    """
    Loads json file.

    :param data: json file.
    :return: Dictionary with the information contained in the json file.
    """
    data_dict = json.loads(data)

    return data_dict


def closest(ts: pd.Series, time_stamp: pd.Timestamp) -> float:
    """
    Get the value from a time series from the one stored in the closest time stamp.

    :param ts: Time series.
    :param time_stamp: Time stamp.
    :return: Value
    """
    return ts.iloc[ts.index.get_loc(time_stamp, method='ffill')]


def bernoulli_distribution(beta):
    """

    :param beta:
    :return:
    """
    return np.random.binomial(1, p=beta)


def create_csv(data: dict = None, output_path: str = None, name: str = None, direct: bool = False,
               from_df: bool = False):
    """
    Creates csv files from a dictionary of pandas series or pandas dataframes.

    :param data: Dictionary with pandas series and/or pandas dataframes.
    :param output_path: Path to store the produced csv.
    """
    if not from_df:
        if not direct:
            for key in data:
                data[key].to_csv(os.path.join(output_path, '{}.csv'.format(key)), header=False)

        else:
            df = pd.DataFrame.from_dict(data, orient='index')
            df.to_csv(os.path.join(output_path, '{}.csv'.format(name)), header=False)

    else:
        data.to_csv(os.path.join(output_path, '{}.csv'.format(name)), header=True)


def create_json(data: dict = None, output_path: str = None, name: str = None, direct: bool = False):
    """
    Creates a json file from a dictionary of pandas series or pandas dataframes.

    :param data: Dictionary with pandas series and/or pandas dataframes.
    :param output_path: Complete output file_path
    """
    if not direct:
        dict_to_jason = dict()
        for key in data:
            dict_to_jason[key] = list(data[key])
    else:
        dict_to_jason = data

    with open(os.path.join(output_path, '{}.json'.format(name)), 'w') as outfile:
        json.dump(dict_to_jason, outfile, indent=4)


def read_dict_objects(dict_objects):
    """
    Reads a dictionary of stored instances, and returns a dataframe with the information ready to be used.

    :param dict_objects: Dictionary with instances.
    :return: Dataframe with the information retrieved from the instances.
    """
    df = pd.DataFrame()
    for obj, data in dict_objects.items():
        df[obj.name] = data

    return df


def create_colours(number_of_colours):
    """
    Creates a list of colours.

    :param number_of_colours:
    :return:
    """
    cm_subsection = np.linspace(.6, 1., number_of_colours)
    colours = [cm.Blues(x) for x in cm_subsection]

    return colours
