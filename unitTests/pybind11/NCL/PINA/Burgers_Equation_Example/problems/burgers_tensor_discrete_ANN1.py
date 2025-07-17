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

noc=0

for i in range(Tmax):
   time = str(i)
   #a.exportU(".",time,"U") 
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

print("U_Mat shape is ====================" , U_Mat.shape)
    
U, S, Vh = np.linalg.svd(U_Mat, full_matrices=True)
Modes = U[:,0:No_Modes]
q_Mat = np.load("q_val_new.npy")


output_data = U_Mat.transpose()
#max_val = max(output_data.reshape(-1,1))
#print("max_val====",max_val)
output_data = output_data
print("The shape of q", output_data.shape)
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
        for i in range(1,Tmax,4):
        #for i in range(1):
          q = outputs_numpy[i-1,:].reshape(-1,1)
          U = np.matmul(Modes,q)
          a.setU(U)
          a.setPrevU()
          a.updatephi()   
          A = a.get_system_matrix(U)
          A_array = A.toarray()
          phi_transpose_A = np.matmul(Modes.transpose(),A_array)
          AR = np.matmul(phi_transpose_A,Modes)
          b = a.get_rhs(U) 
          bR =  np.matmul(Modes.transpose(),b)
          AR_tensor = torch.from_numpy(AR).float()
          bR_tensor = torch.from_numpy(bR).float().reshape(-1,1)
          
          if (i == 1):
            Residual_Physics =  (torch.matmul(AR_tensor,output_[i,:].reshape(-1,1)) - bR_tensor)
          else:
            new_tensor = (torch.matmul(AR_tensor,output_[i,:].reshape(-1,1)) - bR_tensor)
            Residual_Physics = torch.cat((Residual_Physics,new_tensor), dim = 0)
        return 1e2*Residual_Physics
    
    conditions = {
        # 't0':Condition(Span({'t': 0}), initial_condition),
        'D': Condition(Span({'t': input_tensor.reshape(-1,1)}), [burger_equation]),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
