import os
from utils.typings import Errors, PersonalisationMethod
from experiment.AggregatorConfig import AggregatorConfig
#from aggregators.FedMGDAPlusPlus import FedMGDAPlusPlusAggregator
from aggregators.ModAFA import ModAFAAggregator
from aggregators.FedAvg import FedAvgAggregator

from torch import nn, Tensor
from client import Client
from logger import logPrint
from typing import List, Tuple, Type
import torch
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from sklearn.cluster import DBSCAN
import math
import heapq
from utils.PCA import PCA
import hdbscan
import numpy as np
from sklearn.cluster import AgglomerativeClustering





class AgglomerativeAggregator(Aggregator):
    """
    Personalised and Adaptive Dimension Reducing Aggregator.

    Performs PCA on models' weights.

    Uses K-Means clustering with 2 aggregation steps on the PCA transformed data.

    Adaptively decides which clusters should aggregate with each other based on either distance or Cosine similarity.

    NOTE: 10 Epochs per round .
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients: bool = False,
    ):
        super().__init__(clients, model, config, useAsyncClients)

        self.config = config

        
        
        self.internalAggregator: Aggregator = None
        self.externalAggregator: Aggregator = None

        self.blocked_ps = []

        self.personalisation: PersonalisationMethod = config.personalisation
        
        
   
    def trainAndTest(self, testDataset: DatasetInterface) -> Errors:
        roundsError = Errors(torch.zeros(self.config.rounds))
        
        for r in range(self.config.rounds):
            logPrint("Round... ", r)

            # For the initial round everything is as normal.
            # After that, the clients only get the model from their associated cluster
            if r == 0:
                self._shareModelAndTrainOnClients()
            else:
                self._shareModelAndTrainOnClients(self.cluster_centres, self.cluster_labels)

            models = self._retrieveClientModelsDict()
            

            # Perform Clustering
            with torch.no_grad():
                
                self.generate_cluster_centres(models)

                # Assume p value is based on size of cluster
                best_models, ps, indices = self._use_most_similar_clusters()
                conc_ps = [ps[i] for i in indices]
                conc_ps = [p / sum(conc_ps) for p in conc_ps]

                concentrated = self.externalAggregator.aggregate(
                    [FakeClient(p, i) for (i, p) in enumerate(conc_ps)], best_models
                )

                # Update the "best" clusters with a more general model
                for i in range(len(self.cluster_centres)):
                    if i in indices:
                        self.cluster_centres[i] = concentrated

                if self.personalisation == PersonalisationMethod.SELECTIVE:
                    self.model = concentrated
                    roundsError[r] = self.test(testDataset)
            
        return roundsError

                
        
    def _init_aggregators(self, internal: Type[Aggregator], external: Type[Aggregator]) -> None:
        """
        Initialise the internal and external aggregators for access to aggregate function.
        """
        self.internalAggregator = internal(self.clients, self.model, self.config)
        self.externalAggregator = external(self.clients, self.model, self.config)

    def _gen_cluster_centre(self, indices: List[int], models: List[nn.Module]) -> nn.Module:
        """
        Takes the average of the clients assigned to each cluster to generate a new centre

        The aggregation method used should be decided prior.
        """
        return self.internalAggregator.aggregate(
            [self.clients[i] for i in indices], [models[i] for i in indices]
        )

    def _generate_weights(self, models: List[nn.Module]) -> List[Tensor]:
        """
        Converts each model into a tensor of its parameter weights.
        """
        X = [] 
        
        for model in models:
            X.append(self._generate_coords(model))
            

        return X

    def _generate_coords(self, model: nn.Module) -> Tensor:
        """
        Converts the model into a flattened torch tensor representing the model's parameters.
        """
        coords = torch.tensor([]).to(self.device)

        for param in model.parameters():
            coords = torch.cat((coords, param.data.view(-1)))
        
        

        return coords

    def generate_cluster_centres(self, models: List[nn.Module]) -> None:
        """
        Performs PCA with chosen dimension on the model weights

        Uses the PCA transform to perform Agglomerative clustering to get associated labels.

        Aggregate models within clusters to generate cluster centres.
        """

        X = self._generate_weights(models)
        X = [model.tolist() for model in X]
        #pca = PCA.pca2D(X,self.clients)


        

        pca = PCA.pca(X)
        
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=self.config.cluster_distance_threshold,         linkage='complete').fit(pca)
        self.cluster_labels = clusterer.labels_
        



        self.cluster_countagg = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        self.cluster_centres = [None] * self.cluster_countagg
        self.cluster_centres_len = torch.zeros(self.cluster_countagg)

        indices: List[List[int]] = [[] for _ in range(self.cluster_countagg)]
        self.cluster_centres_len.zero_()

        for i, l in enumerate(self.cluster_labels):
            if l != -1:
                self.cluster_centres_len[l] += 1
                indices[l].append(i)

        logPrint(f"Labels: {self.cluster_labels}")
       
        

        self.cluster_centres_len /= len(self.clients)

        for i, ins in enumerate(indices):
             self.cluster_centres[i] = self._gen_cluster_centre(ins, models)
                
                
                
    
    
    
    def _use_most_similar_clustersgood(self, k: int=5) -> Tuple[List[nn.Module], Tensor, List[int]]:
        """
        Uses Cosine similarity to determine the "most similar" clusters and
        selects the top-k most similar clusters for personalization.
        """

        X = self._generate_weights(self.cluster_centres)
        cos = nn.CosineSimilarity(0)

        sims = []
        for i, m1 in enumerate(X):
            sim = []
            for m2 in X:
                sim.append(cos(m1, m2))
            sims.append(sim)

        best_indices = []
        best_val = 0

        # Select the top K largest similarities among models
        for _ in range(k):
            max_val = 0
            max_indices = []
            for i, s in enumerate(sims):
                if i not in best_indices:
                    val = sum([s[j] for j in max_indices])
                    if val > max_val:
                        max_val = val
                        max_indices = [i]
                    elif val == max_val:
                         max_indices.append(i)
            best_indices.extend(max_indices)
            best_val = max_val

        # Normalisation
        ps = Tensor([p / sum(sims[i]) for i in best_indices for p in sims[i]])
        best_models = [self.cluster_centres[i] for i in best_indices]

        # Reconfigure the weights with the size of each cluster
        ps = ps.mul(self.cluster_centres_len)
        ps /= ps.sum()



    
    
        # Personalize the model using the aggregated model from the top-k most similar clusters
        #concentrated = self.externalAggregator.aggregate(
        #[FakeClient(p, i) for (i, p) in enumerate(ps)], best_models
        # )
        #self.model = concentrated

        return best_models, ps, best_indices





    def _use_most_similar_clustersmnisit(self) -> Tuple[List[nn.Module], Tensor, List[int]]:
        """
        Uses Cosine similarity to determine the "most similar" clusters
        """
    
        X = self._generate_weights(self.cluster_centres)
        cos = nn.CosineSimilarity(0)

        sims = []
        for i, m1 in enumerate(X):
            sim = []
            for m2 in X:
                sim.append(cos(m1, m2))
            sims.append(sim)

        best_indices = []
        best_val = 0
    
        # Select all the largest similarities among models
        for i, s in enumerate(sims):
            indices = [j for j, sim in enumerate(s) if sim == max(s)]
            val = sum(s[i] for i in indices)
            if val > best_val:
                best_val = val
                best_indices = indices

        # Normalisation
        ps = Tensor([p / sum(sims[i]) for i in best_indices for p in sims[i]])
        best_models = [self.cluster_centres[i] for i in best_indices]

        # Reconfigure the weights with the size of each cluster
        ps = ps.mul(self.cluster_centres_len)
        ps /= ps.sum()

        return best_models, ps, best_indices
    
    
    


    def _use_most_similar_clusterscifar(self) -> Tuple[List[nn.Module], Tensor, List[int]]:
        """
        Uses Cosine similarity to determin the "most similar" clusters
        
        """
        
        num_to_take = math.floor(self.cluster_countagg / 2)+ 1

        X = self._generate_weights(self.cluster_centres)

        sims = [[] for _ in range(self.cluster_countagg)]
        cos = nn.CosineSimilarity(0)

        for i, m1 in enumerate(X):
            for m2 in X:
                sim = cos(m1, m2)
                sims[i].append(sim)

        best_val = 0
        best_indices: List[int] = []
        besti = -1

        # Uses the group that contains the most similar 3 clusters (K=5) to assign initial weights
        for i, s in enumerate(sims):
            indices = heapq.nlargest(num_to_take, range(len(s)), s.__getitem__)
            val = sum(s[i] for i in indices)
            if val > best_val:
                best_val = val
                best_indices = indices
                besti = i

        # Normalisation
        ps: Tensor = Tensor([p / sum(sims[besti]) for p in sims[besti]])
        best_models = [self.cluster_centres[i] for i in best_indices]

        # If thresholding is being used, threshold the weights based on StD
        if self.config.threshold:
            std = torch.std(ps[ps.nonzero()])
            mean = torch.mean(ps[ps.nonzero()])
            cutoff = mean - std
            ps[ps < cutoff] = 0

        # Reconfigure the weights with the size of each cluster
        ps = ps.mul(self.cluster_centres_len)
        ps /= ps.sum()

        return best_models, ps, best_indices   
    
    
    def _use_most_similar_clusters(self, K=2) -> Tuple[List[nn.Module], Tensor, List[int]]:
        """
        Uses Cosine similarity to determine the "most similar" clusters
        """
        if 0 not in self.cluster_labels:
            num_non_outliers = len(self.cluster_labels)
        else:
            num_non_outliers = (self.cluster_labels != 0).sum()
        num_to_take = min(K, num_non_outliers)
    
        X = self._generate_weights(self.cluster_centres)
        cos = nn.CosineSimilarity(0)

        sims = []
        for i, m1 in enumerate(X):
            sim = []
            for m2 in X:
                sim.append(cos(m1, m2))
            sims.append(sim)

        best_indices = []
        best_val = 0

        # Select the clusters with highest similarities
        for i, s in enumerate(sims):
            indices = heapq.nlargest(num_to_take, range(len(s)), key=s.__getitem__)
            val = sum(s[j] for j in indices)
            if val > best_val:
                best_val = val
                best_indices = indices

        # Normalisation
        ps = Tensor([p / sum(sims[i]) for i in best_indices for p in sims[i]])
        best_models = [self.cluster_centres[i] for i in best_indices]

        # Reconfigure the weights with the size of each cluster
        ps = torch.ones_like(self.cluster_centres_len)
        ps = ps.mul(self.cluster_centres_len)
        ps /= ps.sum()

        return best_models, ps, best_indices



                

    

class FakeClient:
    """
    A fake client for performing external aggregation.

    Useful as setting up a full client is incredibly extra.
    """

    def __init__(self, p: float, id: int):
        self.p = p
        self.id = id



