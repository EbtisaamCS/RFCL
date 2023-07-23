from typing import List, Optional, Tuple

import numpy as np
from datasetLoaders.DatasetLoader import DatasetLoader
from datasetLoaders.DatasetInterface import DatasetInterface
from pandas import DataFrame
from torch import Tensor
from torchvision import transforms, datasets
from logger import logPrint
import torch
from torch.utils.data import DataLoader
class DatasetLoaderCIFAR10(DatasetLoader):
   
    def getDatasets(
        self,
        percUsers: Tensor,
        labels: Tensor,
        size: Optional[Tuple[int, int]] = None,
        nonIID=False,
        alpha=0.1,
        percServerData=0,
    ) -> Tuple[List[DatasetInterface], DatasetInterface]:
        logPrint("Loading Cifar10...")
        self._setRandomSeeds()
        data = self.__loadCIFAR10Data()
        trainDataframe, testDataframe = self._filterDataByLabel(labels, *data)
        
        clientDatasets = self._splitTrainDataIntoClientDatasets(
            percUsers, trainDataframe, self.CIFAR10Dataset, nonIID, alpha
        )
        testDataset = self.CIFAR10Dataset(testDataframe)
        return clientDatasets, testDataset

    @staticmethod
    def __loadCIFAR10Data() -> Tuple[DataFrame, DataFrame]:
        
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

     
        
        trainSet = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
        xTrain = DataLoader(
            trainSet, batch_size=128, shuffle=True)
            
        testSet = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        xTest= DataLoader(
            testSet, batch_size=100, shuffle=False)
            
        xTrain: Tensor = trainSet.data
        xTrain = np.reshape(xTrain,(50000,3072))
        xTrain = xTrain.astype('float32')
        xTrain /= 255
        yTrain = trainSet.targets
        xTest: Tensor = testSet.data
        xTest = np.reshape(xTest,(10000,3072))
        xTest = xTest.astype('float32')
        xTest /= 255
        yTest: np.ndarray = testSet.targets



        trainDataframe = DataFrame(zip(xTrain, yTrain))
        testDataframe = DataFrame(zip(xTest, yTest))
        trainDataframe.columns = testDataframe.columns = ["data", "labels"]

        return trainDataframe, testDataframe

    class CIFAR10Dataset(DatasetInterface):
        def __init__(self, dataframe):
            self.data = torch.stack(
                [torch.from_numpy(data) for data in dataframe["data"].values], dim=0
            )
            super().__init__(dataframe["labels"].values)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def to(self, device):
            self.data = self.data.to(device)
            self.labels = self.labels.to(device)
