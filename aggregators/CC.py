from utils.typings import Errors
from experiment.AggregatorConfig import AggregatorConfig
from torch import nn
from client import Client
from logger import logPrint
from typing import List
import torch
from copy import deepcopy
from aggregators.Aggregator import Aggregator
from datasetLoaders.DatasetInterface import DatasetInterface
import numpy as np


class CenteredClippingAggregator(Aggregator):
    """
    Iterative Centered Clipping Aggregator for Federated Learning.
    """

    def __init__(
        self,
        clients: List[Client],
        model: nn.Module,
        config: AggregatorConfig,
        useAsyncClients=False,
        initial_tau=1e-8,
        threshold=1e-8,
        num_iterations=10,
        center_weight=0.5,
    ):
        super().__init__(clients, model, config, useAsyncClients)
        self.initial_tau = initial_tau
        self.threshold = threshold
        self.num_iterations = num_iterations
        self.center_weight = center_weight

    def trainAndTest(self, testDataset: DatasetInterface) -> Errors:
        roundsError = Errors(torch.zeros(self.rounds))

        for r in range(self.rounds):
            logPrint("Round... ", r)

            self._shareModelAndTrainOnClients()
            models = self._retrieveClientModelsDict()

            # Merge models
            chosen_clients = [self.clients[i] for i in self.chosen_indices]
            self.model = self.aggregate(chosen_clients, models)

            roundsError[r] = self.test(testDataset)

        return roundsError

    def aggregate(self, clients: List[Client], models: List[nn.Module]) -> nn.Module:
        # We can't do aggregation if there are no models this round
        if len(models) == 0:
            return self.model.to(self.device)

        model = models[0]
        modelCopy = deepcopy(model)

        for name, param in model.named_parameters():
            m = []
            for i in range(len(clients)):
                params2 = models[i].named_parameters()
                dictParams2 = dict(params2)
                m.append(dictParams2[name].data.view(-1).to("cpu").numpy())

            m = np.array(m)
            v = param.data.view(-1).to("cpu").numpy()
            tau = self.initial_tau

            for t in range(self.num_iterations):
                # Calculate centered clipping
                center = np.median(m, axis=0)
                radius = np.percentile(np.abs(m - center), q=75, axis=0)
                upper = center + self.center_weight * radius
                lower = center - self.center_weight * radius
                clipped_m = np.clip(m, lower, upper)

                v_new = v + (1 / len(clients)) * np.sum(clipped_m - v, axis=0)
                denominator = tau / np.abs(clipped_m - v_new)
                denominator[np.isinf(denominator)] = 1e-2  # or some other small value
                clipped_m = v_new + ((clipped_m - v_new) * np.minimum(1, denominator))

                dv = np.abs(v_new - v)
                dtau = np.abs(tau - np.abs(clipped_m - v_new).max())

                if np.all(dv < self.threshold) and np.all(dtau < self.threshold):
                    break

                v = v_new
                tau = dtau
                m = clipped_m

            dictParamsm = dict(modelCopy.named_parameters())
            dictParamsm[name].data.copy_(torch.from_numpy(v_new).view(param.data.shape))

        return modelCopy.to(self.device)


