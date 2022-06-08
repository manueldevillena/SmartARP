from tqdm import tqdm
from smartarp import agents
from smartarp.core import SimulationConfiguration


class System:
    """
    Organise the agents' interactions.
    """
    def __init__(self):

        self.dam = agents.DAM()
        self.tso = agents.TSO()
        self.retailer = agents.ECM()
        self.configuration = None

    def run(self, configuration: SimulationConfiguration):
        """
        Runs the simulation.

        :return:
        """
        self.configuration = configuration
        agents = [self.dam, self.tso, self.retailer]

        for agent in agents:
            agent.initialise(configuration)

        for t in tqdm(configuration.simulation_period()):
            for agent in agents:
                agent.act(t, self)
