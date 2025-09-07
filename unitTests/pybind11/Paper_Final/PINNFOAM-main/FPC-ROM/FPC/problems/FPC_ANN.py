import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span
import numpy
from problems.discrete_solver import Residual_Compute
from scipy.io import savemat
import os
from pina import LabelTensor
import pandas as pd
import random
import numpy as np

output_data = pd.read_csv("Uprofile_final_3input.csv",skiprows = None , header = None)
output_data = output_data.values #[0:1001,:]
output_tensor = (torch.from_numpy(output_data)).float()

input_data = pd.read_csv("Time_profile_final.csv",skiprows = None , header = None)
input_data = input_data.values #[0:2001,1].reshape(-1,1)
input_tensor = (torch.from_numpy(input_data)).float()

list_u = []
no_list = 41
for i in range(no_list):
    list_u.append('u_{}'.format(i))

input_tensor = LabelTensor(input_tensor,['alpha','t'])
# input_tensor = LabelTensor(input_tensor,['t'])
output_tensor = LabelTensor(output_tensor,list_u)

B = torch.from_numpy(np.load("Matrices/B.npy")).float()
C = torch.from_numpy(np.load("Matrices/C.npy")).float()
K = torch.from_numpy(np.load("Matrices/K.npy")).float()
M = torch.from_numpy(np.load("Matrices/M.npy")).float()
P = torch.from_numpy(np.load("Matrices/P.npy")).float()


class FPC2D(TimeDependentProblem):
    
    list_u = []
    no_list = 41
    for i in range(no_list):
        list_u.append('u_{}'.format(i))
    output_variables = list_u
    # output_variables = ['u']
    temporal_domain = Span({'alpha': input_tensor[:,0].reshape(-1,1),'t': input_tensor[:,1].reshape(-1,1)})
    # temporal_domain = Span({'t': input_tensor.reshape(-1,1)})
    
    def __init__(self,ntotal,cut_Eq,cut_Data):
        self.ntotal = ntotal
        self.cut_Eq = cut_Eq
        self.cut_Data = cut_Data

    def rand_choice_integer_Eq(self):
        
        list1= [0,1,2,3,4]
        
        list2=[]
        for i in range(self.cut_Eq):
            r=random.randint(5,self.ntotal-1)
            if r not in list1: list1.append(r)
        for i in list1:
            list2.append(i)
        return np.array(list2)
      
    def rand_choice_integer_Data(self):
        
        list1= [0]
        list2=[]
        for i in range(self.cut_Data):
            r=random.randint(1,self.ntotal-20)
            if r not in list1: list1.append(r)
        for i in list1:
            list2.append(i)
        return np.array(list1)
    
    def FPC_equation(self,input_, output_):            
        Ncoeffu = 31
        Ncoeffp = 10
        du = grad(output_, input_)
        dudt_AD = torch.zeros((output_.shape[0],Ncoeffu))
        # dudt_Num = torch.zeros((output_.shape[0],Ncoeffu))         
        for i in range(Ncoeffu):        
            dudt_AD[:,i] = (du.extract(['du_{}dt'.format(i)]))[:,0] 

        dudt_AD_scaled = 0.1*dudt_AD            
        Temporal_Derivative = torch.mm(M,dudt_AD_scaled.t()).t()        
        mdic = {"M_times_dudt":Temporal_Derivative.clone().detach().numpy()}
        savemat("M_times_dudt.mat", mdic)
                
        for i in range(2,output_.shape[0],100):
        # for i in range(2,1001,100):
            mu = 0.01*input_[i,0]
            # print("the value of mu is ==", mu)            
            coeff_c = output_[i,:]
            Residual_Comp = Residual_Compute(mu,Ncoeffu,Ncoeffp,i,coeff_c,B,C,K,M,P)
            spatial_res = Residual_Comp.Residual_spatial()[:,0]
            resu =  Temporal_Derivative[i,:] - spatial_res.detach()
            resp =  Residual_Comp.P_times_a()            
            res = resu[0:].reshape(-1,1)#torch.cat((resu[1:].reshape(-1,1),resp),dim = 0)#resu[0:].reshape(-1,1)#
            resp =  Residual_Comp.P_times_a().detach()
            # print("The shape of res is ==", res.shape)
            
            if i == 2:
               new_tensor = res
               new_tensor_p = resp
            else:
               new_tensor = torch.cat((new_tensor,res), dim = 0)
               new_tensor_p = torch.cat((new_tensor_p,resp), dim = 0)
               
               
               
        # print("The shape of new_tensor is ===", new_tensor.shape)
        mdic = {'new_tensor':new_tensor, 'new_tensor_p':new_tensor_p}
        
        return mdic
            
        
        
    def FPC_equation_derivative(self,input_, output_):
    
        Ncoeffu = 31
        Ncoeffp = 10          
        dR_dUm1_Mat = torch.zeros(Ncoeffu,Ncoeffu+Ncoeffp)
        dRp_dUm1_Mat = torch.zeros(Ncoeffp,Ncoeffu+Ncoeffp)

        for i in range(2,output_.shape[0],100):
        # for i in range(2,1001,100):        
          coeff_c = output_[i,:].reshape(-1,1)
          mu = 0.01*input_[i,0] 
          Residual_Comp = Residual_Compute(mu,Ncoeffu,Ncoeffp,i,coeff_c,B,C,K,M,P)
          spatial_res = Residual_Comp.Residual_spatial()[:,0]          
          res =  Residual_Comp.Residual_spatial()
          resp =  Residual_Comp.P_times_a()
          
          
          for j in range(Ncoeffu+Ncoeffp): 

            # ---------------------------------------------------------------------------
            # --------------------------up ----------------------------------------------
            # ---------------------------------------------------------------------------
                        
            coeff_c[j,0] = coeff_c[j,0] + 0.0001  
                        
            Residual_Comp = Residual_Compute(mu,Ncoeffu,Ncoeffp,i,coeff_c,B,C,K,M,P) 
            spatial_res = Residual_Comp.Residual_spatial()[:,0]
            resu_p =  spatial_res
            resp_P =  Residual_Comp.P_times_a()[:,0]            
            
            # # ---------------------------------------------------------------------------
            # # --------------------------up ----------------------------------------------
            # # ---------------------------------------------------------------------------

         
            coeff_c[j,0] = coeff_c[j,0] - 0.0002
            
            

            
            Residual_Comp = Residual_Compute(mu,Ncoeffu,Ncoeffp,i,coeff_c,B,C,K,M,P)
            spatial_res = Residual_Comp.Residual_spatial()[:,0]
            resu_m = spatial_res            
            resp_m =  Residual_Comp.P_times_a()[:,0]
            # # compute the dR/du now ?
            
            dR_dUm1_Mat[:,j] = (resu_p-resu_m)/(0.0002)
            dRp_dUm1_Mat[:,j] = (resp_P-resp_m)/(0.0002)           
            
            coeff_c[j,0] = coeff_c[j,0] + 0.0001
            
            # #-----------------------------------------------------------------------------------
            # #----------------------compute the updated time ------------------------------------
            # #-----------------------------------------------------------------------------------

          if (i == 2):
            Residual_derivative_m1 =  dR_dUm1_Mat
            Residualp_derivative_m1 =  dRp_dUm1_Mat
            Res_Mat =  res           
            
          else:
            derivative_m1 =  dR_dUm1_Mat
            derivativep_m1 =  dRp_dUm1_Mat
            Residual_derivative_m1 = torch.cat((Residual_derivative_m1,derivative_m1), dim = 0)
            Residualp_derivative_m1 = torch.cat((Residualp_derivative_m1,derivativep_m1), dim = 0)
            Res_Mat = torch.cat((Res_Mat,res), dim = 0)
        
        mdic = {'Residual_derivative_m1':Residual_derivative_m1, 'Residualp_derivative_m1':Residualp_derivative_m1}
        return mdic    
        
        
        
    
    conditions = {
        'A': Condition(Span({'alpha': input_tensor[:,0].reshape(-1,1),'t': input_tensor[:,1].reshape(-1,1)}), [FPC_equation_derivative]),
        'D': Condition(Span({'alpha': input_tensor[:,0].reshape(-1,1),'t': input_tensor[:,1].reshape(-1,1)}), [FPC_equation]),
        # 'A': Condition(Span({'t': input_tensor.reshape(-1,1)}), [burger_equation_derivative]),
        # 'D': Condition(Span({'t': input_tensor.reshape(-1,1)}), [burger_equation]),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
