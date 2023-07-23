from torch import optim
from utils.typings import Errors, PersonalisationMethod
from datasetLoaders.DatasetInterface import DatasetInterface
from experiment.CustomConfig import CustomConfig
import os
from typing import Callable, Dict, List, NewType, Optional, Tuple, Dict, Type
import json
from loguru import logger

from experiment.DefaultExperimentConfiguration import DefaultExperimentConfiguration
from datasetLoaders.MNIST import DatasetLoaderMNIST
from datasetLoaders.CIFAR10 import DatasetLoaderCIFAR10
from datasetLoaders.FashionMNIST import DatasetLoaderFashionMNIST


from classifiers import MNIST
from classifiers import CIFAR10



from logger import logPrint
from client import Client

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import time
import gc
from torch import cuda, Tensor, nn

from aggregators.Aggregator import Aggregator, allAggregators
from aggregators.ModAFA import ModAFAAggregator
from aggregators.AFA import AFAAggregator
from aggregators.FedMGDAplus import FedMGDAplusAggregator
from aggregators.Median import MedianAggregator
from aggregators.MKrum import MKrumAggregator
from aggregators.CC import CenteredClippingAggregator

from aggregators.RFCL import RFCLAggregator
#from aggregators.RFCL_With_FedAvg_Internal_Aggregator import RFCL_With_FedAvg_Internal_AggAggregator
from aggregators.RFCL_Without_PCA import RFCL_Without_PCAAggregator
from aggregators.KMeans import KMeansAggregator
from aggregators.HDBSCAN import HDBSCANAggregator
from aggregators.Agglomerative import AgglomerativeAggregator





from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('large')




SEED = 0

# Colours used for graphing, add more if necessary
COLOURS: List[str] = [

    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:brown",
    "tab:cyan",
    "tab:purple",
    "tab:pink",
    "tab:gray",
    "tab:brown",
    "tab:gray",
    "chartreuse",
    "saddlebrown",
    "blueviolet",
    "navy",
    "cornflowerblue",
    "thistle",
    "dodgerblue",
    "crimson",
    "darkseagreen",
    "maroon",
    "mediumspringgreen",
    "burlywood",
    "olivedrab",
    "linen",
    "mediumorchid",
    "teal",
    "black",
    "gold",
]
styles: List[str]  = ['o', '-', '-','-','-','-', '-']

def __experimentOnMNIST(
    config: DefaultExperimentConfiguration, title="", filename="", folder="DEFAULT"
) -> Dict[str, Errors]:
    """
    MNIST Experiment with default settings
    """
    dataLoader = DatasetLoaderMNIST().getDatasets
    classifier = MNIST.Classifier
    
    #dataLoader = DatasetLoaderCIFAR10().getDatasets
    #classifier = CIFAR10.Classifier


    #dataLoader = DatasetLoaderFashionMNIST().getDatasets
    #classifier = FashionMNIST.Classifier

    
    return __experimentSetup(config, dataLoader, classifier, title, filename, folder)




def __experimentSetup(
    config: DefaultExperimentConfiguration,
    
    datasetLoader: Callable[
        [Tensor, Tensor, Optional[Tuple[int, int]]], Tuple[List[DatasetInterface], DatasetInterface]
    ],
    classifier,
    title: str = "DEFAULT_TITLE",
    filename: str = "DEFAULT_NAME",
    folder: str = "DEFAULT_FOLDER",
) -> Dict[str, Errors]:
    __setRandomSeeds()
    gc.collect()
    cuda.empty_cache()
    errorsDict: Dict[str, Errors] = {}
    blocked: Dict[str, BlockedLocations] = {}

    for aggregator in config.aggregators:
        name = aggregator.__name__.replace("Aggregator", "")
        name = name.replace("Plus", "+")
        logPrint("TRAINING {}".format(name))
        errors = __runExperiment(
                config, datasetLoader, classifier, aggregator, folder
            )
            
        errorsDict[name] = errors.tolist()
     

    # Writing the blocked lists and errors to json files for later inspection.
    if not os.path.isdir(folder):
        os.makedirs(folder)
    if not os.path.isdir(f"{folder}/json"):
        os.mkdir(f"{folder}/json")
    if not os.path.isdir(f"{folder}/graphs"):
        os.mkdir(f"{folder}/graphs")
    #with open(f"{folder}/json/{filename} blocked.json", "w+") as outfile:
        #json.dump(blocked, outfile)
    with open(f"{folder}/json/{filename} errors (Seed: {SEED}).json", "w+") as outfile:
        json.dump(errorsDict, outfile)
   

    # Plots the individual aggregator errors
    if config.plotResults:
        plt.figure()
        i = 0
        for name, err in errorsDict.items():
            #plt.plot(err.numpy(), styles[i], color=COLOURS[i])
           
            plt.plot(err, color=COLOURS[i])
            i += 1
        plt.legend(errorsDict.keys(),prop = fontP)
        plt.xlabel(f"Rounds - {config.epochs} Epochs per Round")
        plt.ylabel("Error Rate (%)")
        plt.title(title, loc="center", wrap=True)
        #plt.ylim(0.5, 1.0)
        plt.ylim(0, 1.0)
        plt.savefig(f"{folder}/graphs/{filename}.pdf",bbox_inches="tight", pad_inches=4.3)

        
    return errorsDict


def __runExperiment(
    config: DefaultExperimentConfiguration,
    datasetLoader: Callable[
        [Tensor, Tensor, Optional[Tuple[int, int]]], Tuple[List[DatasetInterface], DatasetInterface]
    ],
    classifier: nn.Module,
    agg: Type[Aggregator],
    folder: str = "test",
) -> Tuple[Errors]:
    """
    Sets up the experiment to be run.

    Initialises each aggregator appropriately
    """
    serverDataSize = config.serverData
   
    if not agg.requiresData():
        print("Type of agg:", type(agg))
        print("agg:", agg)
        serverDataSize = 0

    trainDatasets, testDataset = datasetLoader(
        config.percUsers,
        config.labels,
        config.datasetSize,
        config.nonIID,
        config.alphaDirichlet,
    )
    
   

    clientPartitions = torch.stack([torch.bincount(t.labels, minlength=10) for t in trainDatasets])
    
    logPrint(
        f"Client data partition (alpha={config.alphaDirichlet}, percentage on server: {100*serverDataSize:.2f}%)"
   )
    logPrint(f"Data per client: {clientPartitions.sum(dim=1)}")
    logPrint(f"Number of samples per class for each client: \n{clientPartitions}")
    plt.rcParams.update({'font.size': 18})
    

   
   
    plt.show()
    
    clients = __initClients(config, trainDatasets)
    # Requires model input size update due to dataset generalisation and categorisation
    
    model = classifier().to(config.aggregatorConfig.device)
    name = agg.__name__.replace("Aggregator", "")

    aggregator = agg(clients, model, config.aggregatorConfig)

  
    if isinstance(aggregator, AFAAggregator):
        aggregator.xi = config.aggregatorConfig.xi
        aggregator.deltaXi = config.aggregatorConfig.deltaXi
    
    #elif isinstance(aggregator, RFCL_With_FedAvg_Internal_AggAggregator):
        #aggregator._init_aggregatorsfed(config.internalfedAggregator, config.externalfedAggregator)
       
        
    elif isinstance(aggregator, RFCL_Without_PCAAggregator):
        aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
    
    
    elif isinstance(aggregator, RFCLAggregator):
        aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
        
        
    elif isinstance(aggregator, KMeansAggregator):
        aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
        
    elif isinstance(aggregator, HDBSCANAggregator):
        aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
   
    elif isinstance(aggregator, AgglomerativeAggregator):
         aggregator._init_aggregators(config.internalAggregator, config.externalAggregator)
  
  
    


    errors: Errors = aggregator.trainAndTest(testDataset)
           

    return errors


def __initClients(
    config: DefaultExperimentConfiguration,
    trainDatasets: List[DatasetInterface],
) -> List[Client]:
    """
    Initialises each client with their datasets, weights and whether they are not benign
    """
    usersNo = config.percUsers.size(0)
    p0 = 1 / usersNo
    logPrint("Creating clients...")
    clients: List[Client] = []
    for i in range(usersNo):
        clients.append(
            Client(
                idx=i,
                trainDataset=trainDatasets[i],
                epochs=config.epochs,
                batchSize=config.batchSize,
                learningRate=config.learningRate,
                p=p0,
                alpha=config.alpha,
                beta=config.beta,
                Loss=config.Loss,
                Optimizer=config.Optimizer,
                device=config.aggregatorConfig.device,
                #needNormalization=config.needNormalization,
            )
        )

    nTrain = sum([client.n for client in clients])
    # Weight the value of the update of each user according to the number of training data points
    for client in clients:
        client.p = client.n / nTrain

    # Create malicious (byzantine) and faulty users
    for client in clients:
        if client.id in config.faulty:
            client.byz = True
            logPrint("User", client.id, "is faulty.")
        if client.id in config.malicious:
            client.flip = True
            logPrint("User", client.id, "is malicious.")
            #client.trainDataset.zeroLabels()
            client.trainDataset.setLabels(6)
       
    return clients


def __setRandomSeeds(seed=SEED) -> None:
    """
    Sets random seeds for all of the relevant modules.

    Ensures consistent and deterministic results from experiments.
    """
    print(f"Setting seeds to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed(seed)


def experiment(exp: Callable[[], None]):
    """
    Decorator for experiments so that time can be known and seeds can be set

    Logger catch is set for better error catching and printing but is not necessary
    """

    @logger.catch
    def decorator():
        __setRandomSeeds()
        logPrint("Experiment {} began.".format(exp.__name__))
        begin = time.time()
        exp()
        end = time.time()
        logPrint("Experiment {} took {}".format(exp.__name__, end - begin))

    return decorator


@experiment
def program() -> None:
    """
    Main program for running the experiments that you want run.
    """
    config = CustomConfig()


    for attackName in config.scenario_conversion():
        errors = __experimentOnMNIST(
            config,
            title=f"",
            filename=f"{attackName}",
            folder=f"test",
        )


# Running the program here
program()


