import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span
import numpy
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
No_Modes = 400
U = a.getU()
print("The size of U is =====", U.size)
t = np.linspace(0, 0.2, 200).reshape(-1,1)
U_Mat = []

noc=0

for i in range(Tmax):
   time = str(i)
   a.exportU(".",time,"U") 
   print("i ====", i)
   U_Mat.append(U)
   for j in range(noc+1):
      A = a.get_system_matrix(U)
      A_array = A.toarray()
      b = a.get_rhs(U)
      U = linalg.solve(A_array, b)
      a.setU(U)
   a.setPrevU()
   a.updatephi()
   print("U is ======", U)

U_Mat = np.column_stack(U_Mat)
np.save("U_Mat_FOM.npy",U_Mat)

print("U_Mat shape is ====================" , U_Mat.shape)
    
"""
U, S, Vh = np.linalg.svd(U_Mat, full_matrices=True)
Modes = U[:,0:No_Modes]
q_Mat = np.load("q_val_new.npy")
"""

output_data = U_Mat.transpose()
#max_val = max(output_data.reshape(-1,1))
#print("max_val====",max_val)
output_data = output_data[:,0:No_Modes]
print("The shape of q", output_data.shape)
output_tensor = (torch.from_numpy(output_data)).float()
t_max = np.max(t)
input_data = t*10/t_max 
input_tensor = (torch.from_numpy(input_data)).float()

list_u = []
no_list = No_Modes
for i in range(no_list):
    list_u.append('u_{}'.format(i))

input_tensor = LabelTensor(input_tensor,['t'])
output_tensor = LabelTensor(output_tensor,list_u)


class NTE(TimeDependentProblem):
    
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
    
    def NTE_equation(self,input_, output_):    
        output_ = output_
        outputs_copy = output_.clone()
        outputs_numpy = outputs_copy.detach().numpy()
        U_z = np.zeros((Tmax,400))
        U_y = outputs_numpy
        U_all = [outputs_numpy,U_y,U_z]
        U_stacked = np.hstack(U_all)
        for i in range(0,Tmax-1,10):
        #for i in range(1):
          U = U_stacked[i,:].reshape(-1,1)
          a.setU(U)
          a.setPrevU()
          a.updatephi()   
          A = a.get_system_matrix(U)
          A_array = A.toarray()
          b = a.get_rhs(U) 
          AR_tensor = torch.from_numpy(A_array).float()
          bR_tensor = torch.from_numpy(b).float().reshape(-1,1)
          
          if (i == 0):
            Residual_Physics =  (torch.matmul(AR_tensor[0:400,0:400],output_[i+1,:].reshape(-1,1)) - bR_tensor[0:400])
          else:
            new_tensor = (torch.matmul(AR_tensor[0:400,0:400],output_[i+1,:].reshape(-1,1)) - bR_tensor[0:400])
            Residual_Physics = torch.cat((Residual_Physics,new_tensor), dim = 0)
        
        return Residual_Physics
    
    def NTE_equation_derivative(self,input_, output_):
        output_ = output_
        outputs_copy = output_.clone().detach()
        outputs_numpy = outputs_copy.numpy()
        U_z = np.zeros((Tmax,400))
        U_y = outputs_numpy
        U_all = [outputs_numpy,U_y,U_z]
        U_stacked = np.hstack(U_all)        
        dR_dUm1_Mat = torch.zeros(output_.size(1),output_.size(1))
        dR_dU_Mat = torch.zeros(output_.size(1),output_.size(1))  

        for i in range(0,Tmax-1,10):
          U = U_stacked[i,:].reshape(-1,1)
          Up = U
          Um = U
          for j in range(output_.size(1)): 


            # ---------------------------------------------------------------------------
            # --------------------------up ----------------------------------------------
            # ---------------------------------------------------------------------------
            
            Up[j,0] = Up[j,0] + 0.00001
            a.setU(Up)
            a.setPrevU()
            a.updatephi() 
            #A = a.get_system_matrix(Up)
            #A_array = A.toarray()
            b = a.get_rhs(Up)
            #AR_tensor = torch.from_numpy(A_array).float()
            bR_tensor = torch.from_numpy(b).float().reshape(-1,1)
            R_Up = -bR_tensor[0:400] #(torch.matmul(AR_tensor[0:400,0:400],outputs_copy[i+1,:].reshape(-1,1)))-bR_tensor[0:400]            
            
            # ---------------------------------------------------------------------------
            # --------------------------up ----------------------------------------------
            # ---------------------------------------------------------------------------

         
            Um[j,0] = Um[j,0] - 0.00002
            a.setU(Um)
            a.setPrevU()
            a.updatephi() 
            #A = a.get_system_matrix(Um)
            #A_array = A.toarray()
            b = a.get_rhs(Um)
            #AR_tensor = torch.from_numpy(A_array).float()
            bR_tensor = torch.from_numpy(b).float().reshape(-1,1)
            R_Um = -bR_tensor[0:400] #(torch.matmul(AR_tensor[0:400,0:400],outputs_copy[i+1,:].reshape(-1,1)))-bR_tensor[0:400]   

            # compute the dR/du now ?
            
            dR_dUm1_Mat[:,j] = (R_Up[:,0]-R_Um[:,0])/(0.00002)
            Um[j,0] = Um[j,0] + 0.00001

            #-----------------------------------------------------------------------------------
            #----------------------compute the updated time ------------------------------------
            #-----------------------------------------------------------------------------------

          a.setU(U)
          a.setPrevU()
          a.updatephi() 
          #dR_dU_Mat_numpy = a.get_system_matrix(U).toarray()
          #dR_dU_Mat = torch.from_numpy(dR_dU_Mat_numpy).float()[0:400,0:400]

          if (i == 0):
            Residual_derivative_m1 =  dR_dUm1_Mat #torch.mm(dR_dU_Mat_Tensor,output_[i+1,:].reshape(-1,1)) + torch.mm(dR_dUm1_Mat,output_[i,:].reshape(-1,1)) 
            #Residual_derivative    =  dR_dU_Mat         
          else:
            derivative_m1 =  dR_dUm1_Mat#torch.mm(dR_dU_Mat_Tensor,output_[i+1,:].reshape(-1,1)) + torch.mm(dR_dUm1_Mat,output_[i,:].reshape(-1,1))
            #derivative = dR_dU_Mat
            Residual_derivative_m1 = torch.cat((Residual_derivative_m1,derivative_m1), dim = 0)
            #Residual_derivative = torch.cat((Residual_derivative,derivative), dim = 0)
        
        mdic = {'Residual_derivative_m1':Residual_derivative_m1}#,'Residual_derivative':Residual_derivative}
        return mdic
            
        
                                             
    conditions = {
        'A': Condition(Span({'t': input_tensor.reshape(-1,1)}), [NTE_equation_derivative]),
        'D': Condition(Span({'t': input_tensor.reshape(-1,1)}), [NTE_equation]),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
