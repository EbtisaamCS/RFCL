from utils.typings import AttacksType
from aggregators.Aggregator import allAggregators
from typing import List
import torch
from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
# Naked imports for allAggregators function
from aggregators. ModAFA import ModAFAAggregator
from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAplus import FedMGDAplusAggregator
#from aggregators.FedMGDAplusplus import FedMGDAplusplusAggregator
from aggregators.FedAvg import FedAvgAggregator
from aggregators.RFCL import RFCLAggregator
#from aggregators.Clustering import ClusteringAggregator

from aggregators.CC import CenteredClippingAggregator
from aggregators.Median import MedianAggregator
from aggregators.MKrum import MKrumAggregator
#from aggregators.RFCL_With_FedAvg_Internal_Aggregator import RFCL_With_FedAvg_Internal_AggAggregator
from aggregators.RFCL_Without_PCA import RFCL_Without_PCAAggregator
from aggregators.KMeans import KMeansAggregator
from aggregators.HDBSCAN import HDBSCANAggregator
from aggregators.Agglomerative import AgglomerativeAggregator



#from aggregators.FedAvg import FAAggregator

class CustomConfig(DefaultExperimentConfiguration):
    def __init__(self):
        super().__init__()

        self.nonIID = True
        self.alphaDirichlet = 0.1  # For sampling
        self.serverData = 1.0 / 6
        # self.aggregatorConfig.rounds = 10

        if self.nonIID:
            iidString = f"non-IID alpha={self.alphaDirichlet}"
        else:
            iidString = "IID"
        
       
        experimentString = f""

        self.scenarios: AttacksType = [
             #([], [], [], f"No Attacks, {iidString} "),
             #([2], [], [], f"1 Byzantine Attack {iidString} {experimentString}"),
            # ([2, 5], [], [], f" 2 Byzantine Attack{iidString} {experimentString}"),
       #([2, 5, 8], [], [], f"3 SF 24,6 FashionMNIST Byzantine Attacks {iidString}{experimentString} "),
             #([2, 5, 8, 11], [], [], f"4 Byzantine Attack {iidString} {experimentString}"),
             #([2, 5, 8, 11, 14], [], [], f"5 Byzantine Attacks {iidString} {experimentString}"),
       #([2, 5, 8, 11, 14, 17], [], [], f"6 SF 21,9  Fashion MNIST Attacks  {iidString}{experimentString}"),
            #([2, 5, 8, 11, 14, 17, 20], [], [], f"7 Byzantine Attacks {iidString} {experimentString}"),
            #([2, 5, 8, 11, 14, 17, 20, 23], [], [], f"faulty_8 {iidString} {experimentString}"),
       #([5,8, 11,14, 17, 20, 23, 26,29], [], [], f"9 SF 21,9 Fashion MNIST Byzantine Attacks  {iidString} {experimentString}"),
            
            #([2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [],   [], f"10 Byzantine Attacks {iidString}",),
             #([1,2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], [], f"11 IPM MNIST Byzantine Attacks {iidString} {experimentString}"),
     ([1,2, 4,5, 8, 11, 14, 17, 20, 23, 26, 29], [], [], f"12 SF 18,12 k2  FashionMNIST  Byzantine Attacks {iidString} {experimentString}"),
             #([1,2, 4,5, 7,8, 11, 14, 17, 20, 23, 26, 29], [], [], f"13 Byzantine Attacks  {iidString} {experimentString}"),
        #([1,2, 4,5, 7,8, 10,11, 14, 17, 20, 23, 26, 29], [], [], f"14 IPM 16,14 k2  FashionMNIST {iidString} {experimentString}"),
            
      #([1,2, 4,5, 7,8, 10,11, 13,14, 17,19, 20, 23, 26,], [], [], f"15 SF 18,12 k2 FashionMNIST  Byzantine Attacks {iidString} {experimentString}"),
            #([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 20, 23, 26, 29], [], [], f"16  epsilon5.5 Cifar ALIE Byzantine Attacks{iidString} {experimentString}"),
       #([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 23, 26, 29], [], [], f"17 SF 18,12K2 FashionMNIST Byzantine Attacks {iidString} {experimentString}"),
     #([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 26, 29], [], [], f"18,SF 16,14 K2 FashionMNIST 15,15 com  Byzantine Attacks {iidString} {experimentString}"),
            #([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 29], [], [], f"faulty_19 {iidString} {experimentString}"),
             #([1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 28,29], [], [], f"20 Byzantine Attacks {iidString} {experimentString}"),
             #([1,2,3, 4,5,6, 7,8, 10,11,12, 13,14, 16,17,18, 19,20,21, 22,23,24, 25,26,27, 28,29,30], [], [], f"30 Byzantine Attacks {iidString} {experimentString}"),
            
            
        #([], [], [2], f"1 A Little Is Enough Attack {iidString} {experimentString}"),
             
        #([], [], [2, 5, 8], f"3 A Little Is Enough Attack {iidString}{experimentString} "),
            
        #([], [], [2, 5, 8, 11, 14, 17], f"6 A Little Is Enough Attack {iidString}{experimentString}"),
           
        #([], [], [5,8, 11,14, 17, 20, 23, 26,29], f"9 A Little Is Enough Attack {iidString} {experimentString}"),
            
        #([], [], [1,2, 4,5, 8, 11, 14, 17, 20, 23, 26, 29], f"12 A Little Is Enough Attack {iidString} {experimentString}"),
            
        #([], [], [1,2, 4,5, 7,8, 10,11, 13,14, 17,19, 20, 23, 26,], f"15 A Little Is Enough Attack {iidString} {experimentString}"),
             
        #([], [], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 26, 29], f"18 A Little Is Enough Attack{iidString} {experimentString}"),
            
        #([], [], [2], f"1 Inner Product Manipulation Attack {iidString} {experimentString}"),
             
        #([], [], [2, 5, 8], f"3 Inner Product Manipulation Attack {iidString}{experimentString} "),
            
        #([], [], [2, 5, 8, 11, 14, 17], f"6 Inner Product Manipulation Attack {iidString}{experimentString}"),
           
        #([], [], [5,8, 11,14, 17, 20, 23, 26,29], f"9 Inner Product Manipulation Attack {iidString} {experimentString}"),
            
        #([], [], [1,2, 4,5, 8, 11, 14, 17, 20, 23, 26, 29], f"12 Inner Product Manipulation Attack {iidString} {experimentString}"),
            
        #([], [], [1,2, 4,5, 7,8, 10,11, 13,14, 17,19, 20, 23, 26,], f"15 Inner Product Manipulation Attack{iidString} {experimentString}"),
             
        #([], [], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 26, 29], f"18 Inner Product Manipulation Attack{iidString} {experimentString}"),
           
            
            
            
            
             #([], [2, ], [], f"1 Label Flipping Attack {iidString} {experimentString}"),
             #([], [2, 5], [], f"2 Label Flipping Attacks {iidString} {experimentString}"),
       # ([], [10,16,20], [], f"3 21,9 Label Flipping Attacks {iidString} {experimentString} "),
           #([], [2, 5, 8, 11], [], f"mal_4 {iidString} {experimentString}"),
             #([], [2, 5, 8, 11, 14], [], f"5 Label Flipping Attacks {iidString} {experimentString}"),
       # ([], [10, 16,20, 23,26,29], [], f"mal_6 21,9 {iidString} {experimentString}"),
            #([], [2, 5, 8, 11, 14, 17, 20], [], f"7 Label Flipping Attacks  {iidString} {experimentString}"),
             #([], [2, 5, 8, 11, 14, 17, 20, 23], [], f"mal_8 {iidString} {experimentString}"),
      #([], [ 10,11,14, 16, 17, 20, 23, 26,29], [], f"mal_9, 15,15 {iidString} {experimentString}"),
        #( [], [2, 5, 8, 11, 14, 17, 20, 23, 26, 29],    [],     f"10 Label Flipping Attacks {iidString} " ),
             #([], [1,2, 5, 8, 11, 14, 17, 20, 23, 26, 29], [], f"mal_11 {iidString} {experimentString}"),

        #([], [2,4,5, 10, 11, 14,16, 17, 20, 23, 26, 29], [], f"12, 5,2 Kmeans Label Flipping Attacks {iidString} {experimentString}"),
            # ([], [1,2, 4,5, 7,8, 11, 14, 17, 20, 23, 26, 29], [], f"mal_13 {iidString} {experimentString}"),
             #([], [1,2, 4,5, 7,8, 10,11, 14, 17, 20, 23, 26, 29], [], f"mal_14 {iidString} {experimentString}"),
       # ([], [2, 4,5, 7,8, 10,11, 13,16, 17, 20, 23, 26,28, 29], [], f"15, 5,2 Kmeans Label Flipping Attacks {iidString} {experimentString}"),
             #([], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 20, 23, 26, 29], [], f"16 Label Flipping Attacks {iidString} {experimentString}"),
             #([], [2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20,22, 23, 26, 29], [], f"17 Label Flipping Attacks  {iidString} {experimentString}"),
       #([], [ 2,4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23,25, 26,29], [], f"mal_18,2,2 Kmeans {iidString} {experimentString}"),
            # ([], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 29], [], f"mal_19 {iidString} {experimentString}"),
            # ([], [1,2, 4,5, 7,8, 10,11, 13,14, 16,17, 19,20, 22,23, 25,26, 28,29], [], f"mal_20 {iidString} {experimentString}"),
            
        ]

        self.percUsers = torch.tensor(PERC_USERS, device=self.aggregatorConfig.device)
        # FedAvg, COMED, MKRUM, FedMGDA+, AFA
        # self.aggregators = [FAAggregator, FedDFAggregator, FedDFmedAggregator, FedBEAggregator, FedBEmedAggregator]
        self.aggregators = [
            #FedAvgAggregator,
            #MedianAggregator,
            #MKrumAggregator,
            #AFAAggregator,
            #FedMGDAplusAggregator,
            #CenteredClippingAggregator,
            ModAFAAggregator,
            #RFCL_Without_PCAAggregator,
            #RFCL_With_FedAvg_Internal_AggAggregator,
            RFCLAggregator,
            #KMeansAggregator,
            #HDBSCANAggregator,
            #AgglomerativeAggregator,
          
        ]

    def scenario_conversion(self):
        """
        Sets the faulty, malicious and free-riding clients appropriately.

        Sets the config's and aggregatorConfig's names to be the attackName.
        """
        for faulty, malicious, freeRider, attackName in self.scenarios:

            self.faulty = faulty
            self.malicious = malicious
            self.freeRiding = freeRider
            self.name = attackName
            self.aggregatorConfig.attackName = attackName

            yield attackName


# Determines how much data each client gets (normalised)
PERC_USERS: List[float] = [
    0.2,
    0.15,
    0.2,
    0.2,
    0.1,
    0.15,
    0.1,
    0.15,
    0.2,
    0.2,
    0.2,
    0.15,
    0.2,
    0.2,
    0.1,
    0.1,
    0.1,
    0.15,
    0.2,
    0.2,
    0.1,
    0.15,
    0.2,
    0.2,
    0.1,
    0.15,
    0.1,
    0.15,
    0.2,
    0.2,
]

