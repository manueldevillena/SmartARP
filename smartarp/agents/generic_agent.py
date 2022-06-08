from abc import ABC, abstractmethod

import pandas as pd


class GenericAgent(ABC):
    """
    Abstract agent of the network.
    """

    def initialise(self, configuration):
        """
        Initialises the agent with default parameters.

        :param configuration: Configuration instance containing all the required inputs to initialise the agents.
        """
        pass

    @abstractmethod
    def act(self, now: pd.Timestamp, system):
        """
        Action of the agent.

        :param now: Time stamp of the simulation process.
        :param system: System instance containing all the information concerning the actions of the agents.
        """
        raise NotImplementedError()
