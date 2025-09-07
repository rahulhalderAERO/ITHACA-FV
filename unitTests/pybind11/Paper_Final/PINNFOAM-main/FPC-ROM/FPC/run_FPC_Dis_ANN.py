import argparse
import torch
from torch.nn import Softplus

from pina import PINN, Plotter, LabelTensor
from pina.model import FeedForward,LSTM
# # from problems.burgers import Burgers1D
from problems.FPC_ANN import FPC2D


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
    ntotal = 3003
    cut_Eqn = 10
    cut_Data = 1
    
    FPC_problem = FPC2D(ntotal,cut_Eqn,cut_Data)
    
    model = FeedForward(
        layers=[124, 100, 80, 64],
        output_variables=FPC_problem.output_variables,
        input_variables=FPC_problem.input_variables,
        extra_features=feat,
    ) #  ,func=Softplus      


    pinn = PINN(
        FPC_problem,
        model,
        lr=0.005,
        error_norm='mse',
        regularizer=0)

    if args.s:
        pinn.span_tensor_given_pts(
            {'n': 100,'variables': 'all'},
            locations=['A','D'])
        pinn.train(5000, 1)
        pinn.save_state('pina.FPC_dis.{}.{}'.format(args.id_run, args.features))        
    else:
        pinn.load_state('pina.FPC_dis.{}.{}'.format(args.id_run, args.features))
        plotter = Plotter()        
        plotter.plot_same_training_test_data(pinn)
        plotter.plot_loss(pinn)

