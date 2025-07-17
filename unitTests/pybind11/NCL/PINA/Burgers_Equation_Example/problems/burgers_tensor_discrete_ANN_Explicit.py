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
import random
import numpy as np
from of_pybind11_system import of_pybind11_system
from scipy import linalg


#Instantiate OF object
a = of_pybind11_system(["."])
dt = 1

# Print the value of U 

#a.printU()

#Get Temperature (T) Field from OF (the memory is shared with OF)
U = a.getU()

#--------------------------------------------------------------------
# LEARN THE Actual SOLUTION 
#-------------------------------------------------------------------
Tmax = 200
No_Modes = 1200
U = a.getU()
print("The size of U is =====", U.size)
t = np.linspace(0, 1, 200).reshape(-1,1)
U_Mat = []

dt = 0.005

for i in range(Tmax):
   time = str(i)
   print("i ====", i)
   a.exportU(".",time,"U")
   U_Mat.append(U)
   U = U + dt*a.getResidual()
   a.setU(U)
   a.setPrevU()
   a.updatephi()
   print("U is ======", U)

U_Mat = np.column_stack(U_Mat)



output_data = U_Mat.transpose()
output_tensor = (torch.from_numpy(output_data)).float()
input_data = t 
input_tensor = (torch.from_numpy(input_data)).float()

list_u = []
no_list = No_Modes
for i in range(no_list):
    list_u.append('u_{}'.format(i))

input_tensor = LabelTensor(input_tensor,['t'])
output_tensor = LabelTensor(output_tensor,list_u)


class Burgers1D(TimeDependentProblem):
    
    list_u = []
    no_list = No_Modes
    for i in range(no_list):
        list_u.append('u_{}'.format(i))
    output_variables = list_u
    temporal_domain = Span({'t': input_tensor.reshape(-1,1)})
    
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
    
    def burger_equation(self,input_, output_):    
        output_ = output_
        outputs_copy = output_.clone()
        outputs_numpy = outputs_copy.detach().numpy()
        for i in range(0,Tmax-1,5):
          U = outputs_numpy[i,:].reshape(-1,1)
          a.setU(U)
          a.setPrevU()
          a.updatephi()   
          Spatial_res = a.getResidual().reshape(-1,1)
          Spatial_res_tensor = (torch.from_numpy(Spatial_res)).float()
          if (i == 0):
            Residual_Physics =  (output_[i+1,:]-output_[i,:]).reshape(-1,1) - dt*Spatial_res_tensor
          else:
            new_tensor =   (output_[i+1,:]-output_[i,:]).reshape(-1,1) - dt*Spatial_res_tensor
            Residual_Physics = torch.cat((Residual_Physics,new_tensor), dim = 0)
        return Residual_Physics

    def burger_equation_derivative(self,input_, output_):
        output_ = output_
        outputs_copy = output_.clone().detach()
        outputs_numpy = outputs_copy.numpy()
        
        
        dR_dUm1_Mat = torch.zeros(output_.size(1),output_.size(1))
        dR_dU_Mat = torch.zeros(output_.size(1),output_.size(1))
        
        for i in range(0,Tmax-1,5):
          U = outputs_numpy[i,:].reshape(-1,1)
          Up = U
          Um = U
          for j in range(output_.size(1)): 


            # ---------------------------------------------------------------------------
            # --------------------------up ----------------------------------------------
            # ---------------------------------------------------------------------------
            Up[j,0] = Up[j,0] + 0.001
            a.setU(Up)
            a.setPrevU()
            a.updatephi() 
            Spatial_res = a.getResidual().reshape(-1,1)
            Spatial_res_tensor = (torch.from_numpy(Spatial_res)).float()  
            R_Up = (outputs_copy[i+1,:]-(torch.from_numpy(Up[:,0])).float()).reshape(-1,1) - dt*Spatial_res_tensor

            # ---------------------------------------------------------------------------
            # --------------------------um-----------------------------------------------
            # ---------------------------------------------------------------------------
            Um[j,0] = Um[j,0] - 0.002
            a.setU(Um)
            a.setPrevU()
            a.updatephi() 
            Spatial_res = a.getResidual().reshape(-1,1)
            Spatial_res_tensor = (torch.from_numpy(Spatial_res)).float()  
            R_Um =  (outputs_copy[i+1,:]-(torch.from_numpy(Um[:,0])).float()).reshape(-1,1) - dt*Spatial_res_tensor

            # compute the dR/du now ?
            
            dR_dUm1_Mat[:,j] = (R_Up[:,0]-R_Um[:,0])/(0.002)
            Um[j,0] = Um[j,0] + 0.001

            #-----------------------------------------------------------------------------------
            #----------------------compute the updated time ------------------------------------
            #-----------------------------------------------------------------------------------

            a.setU(U)
            a.setPrevU()
            a.updatephi() 
            Spatial_res = a.getResidual().reshape(-1,1)
            Spatial_res_tensor = (torch.from_numpy(Spatial_res)).float()





            outputs_copy[i+1,j] = outputs_copy[i+1,j] + 0.001
            R_Up1 = (outputs_copy[i+1,:] -(torch.from_numpy(U[:,0])).float()).reshape(-1,1) - dt*Spatial_res_tensor
            outputs_copy[i+1,j] = outputs_copy[i+1,j] - 0.002
            R_Um1 = (outputs_copy[i+1,:]-(torch.from_numpy(U[:,0])).float()).reshape(-1,1) - dt*Spatial_res_tensor
            dR_dU_Mat[:,j] = (R_Up1[:,0]-R_Um1[:,0])/(0.002)            
            outputs_copy[i+1,j] = outputs_copy[i+1,j] + 0.001


                     
          if (i == 0):
            Residual_derivative_m1 =  dR_dUm1_Mat #torch.mm(dR_dU_Mat_Tensor,output_[i+1,:].reshape(-1,1)) + torch.mm(dR_dUm1_Mat,output_[i,:].reshape(-1,1)) 
            Residual_derivative    =  dR_dU_Mat         
          else:
            derivative_m1 =  dR_dUm1_Mat#torch.mm(dR_dU_Mat_Tensor,output_[i+1,:].reshape(-1,1)) + torch.mm(dR_dUm1_Mat,output_[i,:].reshape(-1,1))
            derivative = dR_dU_Mat
            Residual_derivative_m1 = torch.cat((Residual_derivative_m1,derivative_m1), dim = 0)
            Residual_derivative = torch.cat((Residual_derivative,derivative), dim = 0)
        mdic = {'Residual_derivative_m1':Residual_derivative_m1,'Residual_derivative':Residual_derivative}
        return mdic
        
    
    conditions = {
        'A': Condition(Span({'t': input_tensor.reshape(-1,1)}), [burger_equation_derivative]),
        'D': Condition(Span({'t': input_tensor.reshape(-1,1)}), [burger_equation]),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
