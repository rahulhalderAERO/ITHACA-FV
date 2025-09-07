""" Module for plotting. """
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt
import numpy as np
import torch

from pina import LabelTensor
from pina import PINN
from .problem import SpatialProblem, TimeDependentProblem
from scipy.io import savemat

#from pina.tdproblem1d import TimeDepProblem1D


class Plotter:

    
    def plot_same_training_test_data(self, pinn, components=None, fixed_variables={}, method='contourf',
             res= 1000, filename=None, **kwargs):
        """
        """
        
        for condition_name in pinn.problem.conditions:
            condition = pinn.problem.conditions[condition_name]
            if hasattr(condition, 'output_points'):
                    pts = condition.input_points
                    pts = (pts.to(dtype=pinn.dtype,device=pinn.device))
                    pts.requires_grad_(True)
                    pts.retain_grad()
                    predicted = pinn.model(pts)    
        predicted_output_array = predicted.detach().numpy()
        pts_array = pts.detach().numpy()
        np.save("predicted_output.npy",predicted_output_array) 
        np.save("pts_array.npy",pts_array) 
    
    def plot_loss(self, pinn, label=None, log_scale=True):
        """
        Plot the loss trend

        TODO
        """

        if not label:
            label = str(pinn)

        epochs = list(pinn.history_loss.keys())
        loss = np.array(list(pinn.history_loss.values()))
        # if loss.ndim != 1:
            # loss = loss[:, 0]
        
        mdic_loss = {"epochs":epochs, "loss":loss}
        savemat("burgers_0_100gap.mat", mdic_loss)
        # plt.plot(epochs, loss, label=label)
        # if log_scale:
            # plt.yscale('log')      
