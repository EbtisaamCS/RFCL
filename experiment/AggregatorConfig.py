from utils.typings import PersonalisationMethod
from torch import device, cuda


class AggregatorConfig:
    """
    Configuration for the aggregators.

    Use this for information that you want the aggregator to know about.
    """

    def __init__(self):

        # Total number of training rounds
        self.rounds: int = 5

        self.device = device("cuda" if cuda.is_available() else "cpu")
        #self.device = device("cpu")

        # Name of attack being employed
        self.attackName = ""

        
        
        # FedMGDA+ Parameters:
        self.innerLR: float = 0.1

        #AFA Parameters:
        self.xi: float = 2
        self.deltaXi: float = 0.25
        
        #CC Parameters:
        self.agg_config = {}
        self.agg_config["clip_factor"] = 100.0


         #Clustering Config:
        self.cluster_count: int =23
        self.min_cluster_size=3
        self.hdbscan_min_samples=0.5
        self.cluster_distance_threshold=2.5
        
        
        self.personalisation: PersonalisationMethod = PersonalisationMethod.SELECTIVE
      
       