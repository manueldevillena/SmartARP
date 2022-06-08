import os
import unittest

from smartarp.agents import DAM, Retailer
from smartarp.core import SimulationConfiguration, System
from smartarp.forecast import ShiftForecaster


class TestAgents(unittest.TestCase):
    def setUp(self):
        # Set the working directory to the root.
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def test_dam(self):

        path = 'instances'
        json_file = 'example_test.json'

        system_configuration = SimulationConfiguration(path_input=os.path.join(path, json_file))
        system_configuration.load()
        agent_dam = DAM()
        agent_dam.initialise(system_configuration)

        self.assertAlmostEqual(float(agent_dam.prices['2016-01-01 15:00:00':'2016-01-01 23:00:00'].sum()), 251.91)
        self.assertAlmostEqual(float(agent_dam.prices['2017-07-07 00:00:00':'2017-08-08 23:00:00'].sum()), 25462.27)
        self.assertAlmostEqual(float(agent_dam.prices['2017-01-07 00:00:00':'2017-01-08 23:00:00'].sum()), 2500.72)

    def test_retailer(self):

        path = 'instances'
        json_file = 'example_test.json'

        system_configuration = SimulationConfiguration(path_input=os.path.join(path, json_file))
        system_configuration.load()
        agent_retailer = Retailer()
        agent_retailer.initialise(system_configuration)

        self.assertAlmostEqual(float(agent_retailer.demand_base['2017-03-01 00:00:00':'2017-03-01 23:00:00'].sum()), 1624.930196)
        self.assertAlmostEqual(float(agent_retailer.demand_base['2017-07-07 00:00:00':'2017-08-08 23:00:00'].sum()), 34399.091044)
        self.assertAlmostEqual(float(agent_retailer.demand_base['2017-03-02 00:00:00':'2017-03-02 23:00:00'].sum()), 1343.0430999999999)
        self.assertAlmostEqual(float(agent_retailer.demand_base['2017-01-08 00:00:00':'2017-01-08 23:00:00'].sum()), 885.6783839999999)

    def test_simulation(self):

        path_input = 'instances'
        path_output = 'results'
        json_file = 'example_test.json'

        configuration = SimulationConfiguration(path_input=os.path.join(path_input, json_file), path_output=path_output)
        configuration.load()
        system = System()
        system.run(configuration)
