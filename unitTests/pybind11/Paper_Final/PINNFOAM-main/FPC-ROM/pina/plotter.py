""" Module for plotting. """
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from pina import LabelTensor
from pina import PINN
from .problem import SpatialProblem, TimeDependentProblem
from scipy.io import savemat

#from pina.tdproblem1d import TimeDepProblem1D


class Plotter:

    def plot_samples(self, pinn, variables=None):

        if variables is None:
            variables = pinn.problem.domain.variables
        elif variables == 'spatial':
            variables = pinn.problem.spatial_domain.variables
        elif variables == 'temporal':
            variables = pinn.problem.temporal_domain.variables

        if len(variables) not in [1, 2, 3]:
            raise ValueError

        fig = plt.figure()
        proj = '3d' if len(variables) == 3 else None
        ax = fig.add_subplot(projection=proj)
        for location in pinn.input_pts:
            coords = pinn.input_pts[location].extract(variables).T.detach()
            if coords.shape[0] == 1:  # 1D samples
                ax.plot(coords[0], torch.zeros(coords[0].shape), '.',
                        label=location)
            else:
                ax.plot(*coords, '.', label=location)

        ax.set_xlabel(variables[0])
        try:
            ax.set_ylabel(variables[1])
        except:
            pass

        try:
            ax.set_zlabel(variables[2])
        except:
            pass

        plt.legend()
        plt.show()

    def _1d_plot(self, pts, pred, method, truth_solution=None, **kwargs):
        """
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

        ax.plot(pts, pred.detach(), **kwargs)

        if truth_solution:
            truth_output = truth_solution(pts).float()
            ax.plot(pts, truth_output.detach(), **kwargs)

        plt.xlabel(pts.labels[0])
        plt.ylabel(pred.labels[0])
        plt.show()

    def _2d_plot(self, pts, pred, v, res, method, truth_solution=None,
                 **kwargs):
        """
        """

        grids = [p_.reshape(res, res) for p_ in pts.extract(v).cpu().T]

        pred_output = pred.reshape(res, res)
        if truth_solution:
            truth_output = truth_solution(pts).float().reshape(res, res)
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

            cb = getattr(ax[0], method)(*grids, pred_output.cpu().detach(), **kwargs)
            fig.colorbar(cb, ax=ax[0])
            cb = getattr(ax[1], method)(*grids, truth_output.cpu().detach(), **kwargs)
            fig.colorbar(cb, ax=ax[1])
            cb = getattr(ax[2], method)(*grids,
                                        (truth_output-pred_output).cpu().detach(),
                                        **kwargs)
            fig.colorbar(cb, ax=ax[2])
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            cb = getattr(ax, method)(*grids, pred_output.cpu().detach(), **kwargs)
            fig.colorbar(cb, ax=ax)

    
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
        mdic = {"predicted_output_test_0":predicted_output_array, "pts_array_test_0":pts_array}
        savemat("Output/FPC_Dis_test_0.mat", mdic)
        
    
            
        

    def plot_loss(self, pinn, label = None, log_scale = True):
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
        
        mdic_loss = {"epochs_dis_test_0":epochs, "loss_dis_test_0":loss}
        savemat("Output/FPC_loss_dis_test_0.mat", mdic_loss)
        # plt.plot(epochs, loss, label=label)
        # if log_scale:
            # plt.yscale('log')
