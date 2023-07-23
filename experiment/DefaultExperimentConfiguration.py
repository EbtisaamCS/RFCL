from aggregators.ModAFA import ModAFAAggregator
from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAplus import FedMGDAplusAggregator
from aggregators.RFCL import RFCLAggregator
from aggregators.Median import MedianAggregator
from aggregators.MKrum import MKrumAggregator
from aggregators.CC import CenteredClippingAggregator

from experiment.AggregatorConfig import AggregatorConfig
from aggregators.FedAvg import FedAvgAggregator
#from aggregators.RFCL_With_FedAvg_Internal_Aggregator import RFCL_With_FedAvg_Internal_AggAggregator
from aggregators.RFCL_Without_PCA import RFCL_Without_PCAAggregator
from aggregators.KMeans import KMeansAggregator
from aggregators.HDBSCAN import HDBSCANAggregator
from aggregators.Agglomerative import AgglomerativeAggregator







import torch
from aggregators.Aggregator import Aggregator, allAggregators
from typing import List, Tuple, Type, Union
import torch.optim as optim
import torch.nn as nn


class DefaultExperimentConfiguration:
    """
    Base configuration for the federated learning setup.
    """

    def __init__(self):
        # DEFAULT PARAMETERS
        self.name: str = ""
        

        self.aggregatorConfig = AggregatorConfig()

        # Epochs num locally run by clients before sending back the model update
        self.epochs: int = 10 

        self.batchSize: int = 64 #128  # Local training  batch size
        self.learningRate: float = 0.05 #0.1 
        self.Loss = nn.CrossEntropyLoss
        self.Optimizer: Type[optim.Optimizer] = optim.SGD

        # Big datasets size tuning param: (trainSize, testSize); (None, None) interpreted as full dataset
        self.datasetSize: Tuple[int, int] = (0, 0)

        # Clients setup
        self.percUsers = torch.tensor(
            [0.2, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1]
        )  # Client data partition
        self.labels = torch.tensor(range(10))  # Considered dataset labels
        self.faulty: List[int] = []  # List of noisy clients
        self.malicious: List[int] = []  # List of (malicious) clients with flipped labels

        #AFA Parameters:
        self.alpha: float = 3
        self.beta: float = 3

        self.aggregators: List[Type[Aggregator]] = allAggregators()  # Aggregation strategies

        self.plotResults: bool = True

        # Group-Wise config
        
        #self.internalAggregator: Union[
            #Type[FedAvgAggregator], Type[ModAFAAggregator]]
        #if (self.aggregators==RFCL_With_FedAvgAggregator):
        self.internalfedAggregator=  FedAvgAggregator 
        #self.externalAggregator: Union[
            #Type[FedAvgAggregator] #Type[MKRUMAggregator], Type[COMEDAggregator]
        #] = FAAggregator
        self.externalfedAggregator= FedAvgAggregator
       # elif (self.aggregators==RFCLAggregator):
        self.internalAggregator= ModAFAAggregator # ModAFAAggregator     # MedianAggregator
        self.externalAggregator= FedAvgAggregator

        # Data splitting config
        self.nonIID = True
        self.alphaDirichlet = 0.1  # Parameter for Dirichlet sampling
       