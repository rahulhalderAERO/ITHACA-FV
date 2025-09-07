import argparse
import torch
from torch.nn import Softplus
import numpy as np
from pina import PINN,Plotter,LabelTensor
from pina.model import FeedForward,LSTM
from problems.NTE_ANN import NTE
from of_pybind11_system import of_pybind11_system
import os


#Instantiate OF object
a = of_pybind11_system(["."])


class myFeature(torch.nn.Module):
    """
    Feature: sin(pi*x)
    """
    def __init__(self, idx):
        super(myFeature, self).__init__()
        self.idx = idx

    def forward(self, x):
        return LabelTensor(torch.sin(torch.pi * x.extract(['x'])), ['sin(x)'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help = "number of run", type=int)
    parser.add_argument("features", help = "extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature(0)] if args.features else []
    ntotal = 200
    cut_Eqn = 10
    cut_Data = 2
    
    burgers_problem = NTE(ntotal,cut_Eqn,cut_Data)
    
    model = FeedForward(
        layers=[124,100,80,64],
        output_variables=burgers_problem.output_variables,
        input_variables=burgers_problem.input_variables,
        func=Softplus,
        extra_features=feat,
    )#        

    pinn = PINN(
        burgers_problem,
        model,
        lr=0.006,
        error_norm='mse',
        regularizer=0)

    if args.s:
        pinn.span_tensor_given_pts(
            {'n': 200,'variables': 'all'},
            locations=['A','D'])
        pinn.train(3999, 1)
        pinn.save_state('pina.burger_disdataphysics_0.{}.{}'.format(args.id_run, args.features))
    else:
        U_mat = np.load("U_Mat_FOM.npy").transpose()
        pinn.load_state(f"pina.burger_disdataphysics_0.{args.id_run}.{args.features}")
        plotter = Plotter()
        plotter.plot_same_training_test_data(pinn)
        plotter.plot_loss(pinn)
        predicted_output = np.load("predicted_output.npy")
        print("predicted_output ==", predicted_output)
        U_z = np.zeros((200, 400))
        U_y = predicted_output
        U_all = [predicted_output, U_y, U_z]
        U = np.hstack(U_all)
        Tmax = 200
        pred_base = "Prediction_With_DataPhysics1_0"
        err_base  = "Error_With_DataPhysicscheck1_0"
        for i in range(Tmax):
           print("I am at ========", i)
           time = str(i)
           U_instant = U[i, :].reshape(-1, 1)
           U_error   = np.abs(U[i, :] - U_mat[i, :]).reshape(-1, 1)
           # ensure per-step folders exist (no function)
           pred_dir = os.path.join(pred_base, time)
           err_dir  = os.path.join(err_base,  time)
           os.makedirs(pred_dir, exist_ok=True)
           os.makedirs(err_dir,  exist_ok=True)
           a.setU(U_instant)
           if i != 0:  # keep your original behavior
              a.exportU(".", pred_dir, "U")
              a.setU(U_error)
              a.exportU(".", err_dir, "U")

     


