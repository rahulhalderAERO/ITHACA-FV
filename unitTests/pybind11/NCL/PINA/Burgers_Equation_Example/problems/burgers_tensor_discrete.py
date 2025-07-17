import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span
import numpy
from problems.Burgers_DiscreteTorch_Class import Burgers_Discrete
from scipy.io import savemat
import os
from pina import LabelTensor




class Burgers1D(TimeDependentProblem, SpatialProblem):

    output_variables = ['u']
    spatial_domain = Span({'x': [0, 1]})
    temporal_domain = Span({'t':[0, 0.5]})

    def burger_equation(input_, output_):
        
        nx = 20
        nt = 100
        mu = 0.05
        L = torch.linspace(0, 1, nx).reshape(-1, 1)
        time = torch.linspace(0, 0.5, nt).reshape(-1, 1)
        x_ext = (input_.extract(['x']))
        x_ext_Mat = (torch.reshape(x_ext, (nx, nt)))        
        t_ext = (input_.extract(['t']))
        t_ext_Mat = (torch.reshape(t_ext, (nx, nt)))
        u_ext = (output_.extract(['u']))        
        u_ext_Mat = (torch.reshape(u_ext, (nx, nt))) 
         
        
        x_ext_name = []
        for i in range(nt):
            x_ext_name.append("xcol_{}".format(i))
        t_ext_name = []
        for i in range(nt):
            t_ext_name.append("tcol_{}".format(i))    
        u_ext_name = []
        for i in range(nt):
            u_ext_name.append("ucol_{}".format(i))
        
        x_Mat = LabelTensor(x_ext_Mat, x_ext_name)
        t_Mat = LabelTensor(t_ext_Mat, t_ext_name)
        u_Mat = LabelTensor(u_ext_Mat, u_ext_name)

        
        for i in range(u_ext_Mat.size(1)-1):
            Burgers_Prob = Burgers_Discrete(L,time,mu,u_Mat.extract(['ucol_{}'.format(i)]),u_Mat.extract(['ucol_{}'.format(i+1)]))
            if (i == 0):
                new_tensor = (Burgers_Prob.Burgers_Residual())
            else:
                new_tensor = torch.cat((new_tensor, (Burgers_Prob.Burgers_Residual())), dim = 0)               
        ADis = (new_tensor)
        
        return ADis

    def nil_dirichlet(input_, output_):
        u_expected = 0.0
        return output_.extract(['u']) - u_expected

    def initial_condition(input_, output_):
        u_expected = torch.sin(torch.pi*input_.extract(['x']))
        return output_.extract(['u']) - u_expected

    conditions = {
        'gamma1': Condition(Span({'x': 0, 't': [0, 0.5]}), nil_dirichlet),
        'gamma2': Condition(Span({'x':  1, 't': [0,0.5]}), nil_dirichlet),
        't0': Condition(Span({'x': [0, 1], 't': 0}), initial_condition),
        'D': Condition(Span({'x': [0, 1], 't': [0, 0.5]}), burger_equation),
    }
