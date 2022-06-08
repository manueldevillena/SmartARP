import os
import unittest

import pandas as pd

from smartarp.forecast import ImprovingForecaster


class TestForecast(unittest.TestCase):
    def setUp(self):
        # Set the working directory to the root.
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def test_improving_forecaster(self):
        series = pd.read_csv("data_dummy/load_retailer_total.csv", header=0, index_col=0, usecols=[0, 1],
                             parse_dates=True, infer_datetime_format=True, squeeze=True) / 250.0

        ratio = 0.03
        forecaster = ImprovingForecaster(min_error_ratio=ratio)
        forecaster.train(series)

        # Get series values
        target_time = pd.Timestamp(2017, 1, 2, 21, 0, 0)  # value: 11.004
        real_value = series[target_time]
        day_ahead_value = series[target_time - pd.Timedelta(hours=24)]

        # Check forecasts
        v = forecaster.forecast(target_time, target_time - pd.Timedelta(hours=24))
        self.assertAlmostEqual(v, day_ahead_value)

        v = forecaster.forecast(target_time, target_time)
        self.assertAlmostEqual(v, real_value - ratio * (real_value - day_ahead_value))

        v = forecaster.forecast(target_time, target_time - pd.Timedelta(hours=12))
        self.assertAlmostEqual(v, real_value - (real_value - day_ahead_value) / 2)

