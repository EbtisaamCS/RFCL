import copy
from typing import Optional, Type,List,Tuple, Callable,Any, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch import Tensor, cuda
from torch import nn, Tensor, randn, tensor, device, float64,cuda
from torch.utils.data import DataLoader

from numpy import clip, percentile

from scipy.stats import laplace

from logger import logPrint

import gc

import torch.distributed as dist
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm


class Client:
    """An internal representation of a client"""

    def __init__(
        self,
        epochs,
        batchSize,
        learningRate,
        trainDataset,
        p,
        idx,
        device,
        Optimizer,
        Loss,
        #needNormalization,
        byzantine=None,
        flipping=None,
        freeRiding=False,
        model: Optional[nn.Module] = None,
        alpha=3.0,
        beta=3.0,
    ):

        self.name: str = "client" + str(idx)
        self.device: torch.device = device
        #self.device: torch.device  = device("cuda" if cuda.is_available() else "cpu")
        #self.gradients: list = gradients

        self.model: nn.Module = model
        self.trainDataset = trainDataset
        self.trainDataset.to(device)
        self.dataLoader = DataLoader(self.trainDataset, batch_size=batchSize, shuffle=True)
        self.n: int = len(trainDataset)  # Number of training points provided
        self.p: float = p  # Contribution to the overall model
        self.id: int = idx  # ID for the user
        self.byz: bool = byzantine  # Boolean indicating whether the user is faulty or not
        self.flip: bool = flipping  # Boolean indicating whether the user is malicious or not (label flipping attack)
       
        self.untrainedModel: nn.Module = copy.deepcopy(model).to("cpu") if model else None

        # Used for free-riders delta weights attacks
        self.prev_model: nn.Module = None

        self.opt: optim.Optimizer = None
        self.sim: Tensor = None
        self.loss = None
        self.Loss = Loss
        self.Optimizer: Type[optim.Optimizer] = Optimizer
        self.pEpoch: float = None
        self.badUpdate: bool = False
        self.epochs: int = epochs
        self.batchSize: int = batchSize

        self.learningRate: float = learningRate
        self.momentum: float = 0.9

        # AFA Client params
        self.alpha: float = alpha
        self.beta: float = beta
        self.score: float = alpha / beta
        self.blocked: bool = False


    def updateModel(self, model: nn.Module) -> None:
        """
        Updates the client with the new model and re-initialise the optimiser
        """
        self.prev_model = copy.deepcopy(self.model)
        self.model = model.to(self.device)
        if self.Optimizer == optim.SGD:
            self.opt = self.Optimizer(
                self.model.parameters(), lr=self.learningRate, momentum=self.momentum
            )
        else:
            self.opt = self.Optimizer(self.model.parameters(), lr=self.learningRate)
        self.loss = self.Loss()
        self.untrainedModel = copy.deepcopy(model)
        cuda.empty_cache()

    def trainModel(self):
        """
        Trains the client's model unless the client is a free-rider
        """
       
        self.model = self.model.to(self.device)
        for i in range(self.epochs):
            for x, y in self.dataLoader:
                x = x.to(self.device)
                y = y.to(self.device)
                err, pred = self._trainClassifier(x, y)

        gc.collect()
        cuda.empty_cache()
        self.model = self.model
        return err, pred

    def _trainClassifier(self, x: Tensor, y: Tensor):
        """
        Trains the classifier
        """
        x = x.to(self.device)
        y = y.to(self.device)
        # Reset gradients
        self.opt.zero_grad()
        pred = F.softmax(self.model(x).to(self.device), dim=1)
        err = self.loss(pred, y).to(self.device)
        err.backward()
        # Update optimizer
        self.opt.step()
        return err, pred

    def retrieveModel(self) -> nn.Module:
        """
        Function used by aggregators to retrieve the model from the client
        
        """
       
        if self.byz:
            # Faulty model update
            #self.add_noise_to_gradients()
            self.flip_signs()
            #self.byzantine_attack()
            #self.__manipulateModel()
            #self.ALittleIsEnoughAttack()
            #self.IPMAttack()

        
        return self.model

    def __manipulateModel(self, alpha: int = 20) -> None:
        """
        Function to manipulate the model for byzantine adversaries
        """
        for param in self.model.parameters():
            noise = alpha * torch.randn(param.data.size()).to(self.device)
            param.data.copy_(param.data.to(self.device) + noise)
        
        
    

    def byzantine_attack(self, epsilon: float = 0.5 ):
        """
        This code randomly adds Gaussian noise to half of the model parameters, and flips the sign of the other half. The epsilon argument             determines the magnitude of the Gaussian noise added to the parameters. Note that this function modifies the model parameters in place,         so there is no need to return anything.
        Manipulates the model parameters to simulate Byzantine attacks.

        Args:
        epsilon (float): the magnitude of the perturbation to add to the model parameters.

        Returns:
        None
   
        """
        for param in self.model.parameters():
            if torch.rand(1) < 0.5:
               # Add random noise to half of the parameters
               noise = torch.randn_like(param) * epsilon
               param.data.add_(noise).to(self.device)
            else:
               # Flip the sign of the other half of the parameters
               param.data.mul_(-1)
        
    def flip_signs(self,):
        """
        This function flips the signs of all parameters of the model.
        """
        #This loops through all the parameters of the model.
        for param in self.model.parameters():
        #This multiplies the data of each parameter with -1, effectively flipping the signs of all the parameters.
        #The mul_ method is an in-place multiplication, meaning it modifies the tensor in place.
            param.data.mul_(-1)



            
    

    def add_noise_to_gradients(self,) -> None:
        """
        Generates gradients based on random noise parameters.
        Noise parameters should be tweaked to be more representative of the data used.
        """
        # Get the current model parameters
        model_params = list(self.model.parameters())

        # Compute the perturbation
        perturbation = []
        for param in model_params:
            noise = torch.randn_like(param)  # Generate Gaussian noise with the same shape as the parameter
            noise_norm = torch.norm(noise.view(-1), p=2)  # Compute the norm of the noise
            perturbation.append(20 * noise )  # Scale the noise to have standard deviation 20

        # Apply the perturbation to the model parameters
        for i, param in enumerate(model_params):
            param.data.copy_(param.data + perturbation[i])
    
    
    
        
    def ALittleIsEnoughAttack(self, n=30, m=18, z=None, epsilon: float = 0.5) -> None:
        device = next(self.model.parameters()).device

        # Calculate mean and std over benign updates
        model_params = list(self.model.parameters())
        means, stds = [], []
        
        for param in self.model.parameters():
            if param.grad is not None and param.grad.numel() > 0:
                updates = param.grad.view(param.grad.shape[0], -1)
                mean, std = torch.mean(updates, dim=1), torch.std(updates, dim=1)
                means.append(mean)
                stds.append(std)
        self.benign_mean = means
        self.benign_std = stds

        if z is not None:
            self.z_max = z
        else:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = norm.ppf(cdf_value)
            self.n_good = n - m

        if z is None:
            z = 1.0 

        # Zero the parameter gradients
        self.model.zero_grad()

        # Compute the perturbation
        perturbation = []
        for i, (param, mean, std) in enumerate(zip(self.model.parameters(), self.benign_mean, self.benign_std)):
            delta = torch.randn_like(param.grad.view(param.grad.shape[0], -1))
            perturbed_delta = torch.clamp(delta, -z * float(std[0]), z * float(std[0]))
            lower = self.benign_mean[i] - self.z_max * self.benign_std[i]
            upper = self.benign_mean[i] + self.z_max * self.benign_std[i]
            perturbed_param = param.data.to(device) + epsilon * perturbed_delta.view(param.grad.shape)
            perturbed_param = torch.clamp(perturbed_param, float(lower[0]), float(upper[0]))
            perturbation.append(perturbed_param - param.data.to(device))

            


        # Apply the perturbation to the model parameters
        for i, param in enumerate(model_params):
            param.data.copy_(param.data.to(device) + perturbation[i])


               
    def IPMAttack(self, std_dev: float = 0.5 ) -> None:
        
        """
        Performs an inner product manipulation attack on a model by modifying the
        model's gradients.

        Args:
        model (nn.Module): the PyTorch model to attack.
        epsilon (float): the magnitude of the perturbation to add to the gradients.

        Returns:
        None
        """
        # Get the current model parameters
        model_params = list(self.model.parameters())

        # Calculate the gradients for each batch and accumulate them
        gradients = [torch.zeros_like(param) for param in model_params]

        # Accumulate gradients
        for i, param in enumerate(model_params):
            gradients[i] += param.grad.clone()

        # Compute the inner products of the gradients
        inner_products = [torch.dot(grad.view(-1), param.view(-1)).item() for grad, param in zip(gradients, model_params)]

        # Compute the perturbation
        perturbation = []
        for i, param in enumerate(model_params):
            perturbation.append(std_dev * inner_products[i])

        # Apply the perturbation to the gradients
        for i, param in enumerate(model_params):
            param.data.copy_(param.data.to(self.device) + perturbation[i])
            

 

     

    