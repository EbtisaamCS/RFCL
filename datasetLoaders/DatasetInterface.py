import torch
from torch.utils.data import Dataset


class DatasetInterface(Dataset):
    def __init__(self, labels):
        self.labels = torch.tensor(labels, dtype=torch.long)
         

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        raise Exception("Method should be implemented in subclass.")

    def getInputSize(self):
        raise Exception(
            "Method should be implemented by subclasses where "
            "models requires input size update (based on dataset)."
        )

   
    def setLabels(self, value: int) -> None:
        """
        Sets target labels to the given value
        """
        for index in range(len(self.labels)):
            if self.labels[index] == 0:
                self.labels[index] = value  
        #Sets all labels to the given value
        #self.labels = torch.ones(len(self.labels), dtype=torch.long) * value
        
    def replace_0_with_6(targets, target_set):
        """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
       """
        for idx in range(len(targets)):
            if targets[idx] == 0:
                targets[idx] = 6

        return targets
