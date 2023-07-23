from datasetLoaders.DatasetInterface import DatasetInterface
import os
import random
import re
from typing import List, Tuple, Type
import numpy as np
import pandas as pd
import torch
from torch import Tensor, cuda
from pandas import DataFrame



class DatasetLoader:
    """Parent class used for specifying the data loading workflow"""

    def getDatasets(
        self,
        percUsers: Tensor,
        labels: Tensor,
        size=(None, None),
        nonIID=False,
        alpha=0.1,
        percServerData=0,
    ):
        raise Exception(
            "LoadData method should be override by child class, "
            "specific to the loaded dataset strategy."
        )

    @staticmethod
    def _filterDataByLabel(
        labels: Tensor, trainDataframe: DataFrame, testDataframe: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Creates the train and test dataframes with only the labels specified
        """
        trainDataframe = trainDataframe[trainDataframe["labels"].isin(labels.tolist())]
        testDataframe = testDataframe[testDataframe["labels"].isin(labels.tolist())]
        return trainDataframe, testDataframe

    @staticmethod
    def _splitTrainDataIntoClientDatasets(
        percUsers: Tensor,
        trainDataframe: DataFrame,
        DatasetType: Type[DatasetInterface],
        nonIID,
        alpha,
    ) -> List[DatasetInterface]:
        """
        Splits train dataset into individual datasets for each client.

        Uses percUsers to decide how much data (by %) each client should get.
        """
        DatasetLoader._setRandomSeeds()
        percUsers = percUsers / percUsers.sum()

        if not nonIID:
            dataSplitCount = (percUsers.cpu() * len(trainDataframe)).floor().numpy()
            _, *dataSplitIndex = [
                int(sum(dataSplitCount[range(i)])) for i in range(len(dataSplitCount))
            ]

            # Sample and reset_index shuffles the dataset in-place and resets the index
            trainDataframes: List[DataFrame] = np.split(
                trainDataframe.sample(frac=1).reset_index(drop=True),
                indices_or_sections=dataSplitIndex,
            )

            clientDatasets: List[DatasetInterface] = [
                DatasetType(clientDataframe.reset_index(drop=True))
                for clientDataframe in trainDataframes
            ]

        else:
            # print(f"SPLITTING THE DATA IN A NON-IID MANNER USING ALPHA={alpha}")

            # Split dataframe by label
            gb = trainDataframe.groupby(
                by="labels"
            )  # NOTE: What if the labels aren't called labels? Might need it as input
            label_groups = [gb.get_group(x) for x in gb.groups]

            num_classes = len(label_groups)
            num_clients = len(percUsers)
            p = alpha * np.ones(num_clients)
            percUserPerClass = np.random.dirichlet(p, num_classes)

            clientsData = []
            for i in range(num_classes):
                percUsers = percUserPerClass[i]

                dataSplitCount = np.floor(percUsers * len(label_groups[i]))
                _, *dataSplitIndex = [
                    int(sum(dataSplitCount[:i])) for i in range(len(dataSplitCount))
                ]

                # Sample and reset_index shuffles the dataset in-place and resets the index
                trainDataframes: List[DataFrame] = np.split(
                    label_groups[i].sample(frac=1).reset_index(drop=True),
                    indices_or_sections=dataSplitIndex,
                )

                clientsData.append(trainDataframes)

            clientsDataMerged = []
            for j in range(num_clients):
                tmp = []
                for i in range(num_classes):
                    tmp.append(clientsData[i][j])
                clientsDataMerged.append(pd.concat(tmp, ignore_index=True))

             #for cliData in clientsDataMerged:
             #print(cliData['labels'].value_counts().sort_index())
             #print("length of cliData", len(cliData))

            clientDatasets: List[DatasetInterface] = [
                DatasetType(clientDataframe.reset_index(drop=True))
                for clientDataframe in clientsDataMerged
            ]
        return clientDatasets

    @staticmethod
    def _setRandomSeeds(seed=0) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cuda.manual_seed(seed)

   
    @staticmethod
    def __leastGeneral(map1, map2, domainSize):
        map1Generality = map2Generality = 0
        for col in map1:
            if isinstance(map1[col], str):
                interval = np.array(re.findall(r"\d+.\d+", map1[col]), dtype=np.float)
                map1Generality += (interval[1] - interval[0]) / domainSize[col]

        for col in map2:
            if isinstance(map1[col], str):
                interval = np.array(re.findall(r"\d+.\d+", map2[col]), dtype=np.float)
                map2Generality += (interval[1] - interval[0]) / domainSize[col]

        return map1 if map1Generality <= map2Generality else map2

    @staticmethod
    def __legitMapping(entry, mapping) -> bool:
        for col in mapping:
            if not isinstance(mapping[col], str):
                if entry[col] != mapping[col]:
                    return False
            else:
                interval = np.array(re.findall(r"\d+.\d+", mapping[col]), dtype=np.float)
                if interval[0] < entry[col] or entry[col] >= interval[1]:
                    return False
        return True
