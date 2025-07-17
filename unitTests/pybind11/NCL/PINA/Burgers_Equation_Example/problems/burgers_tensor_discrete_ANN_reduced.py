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
No_Modes = 50
U = a.getU()
print("The size of U is =====", U.size)
t = np.linspace(0, 1, 200).reshape(-1,1)
U_Mat = []

noc=0

for i in range(Tmax):

 if (i == 0):
   
   print("Reduced i is ====", i)
   time = str(i)
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

 else:
   print("Reduced i is ====", i)
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
print("U_Mat shape is ====================" , U_Mat.shape)
U, S, Vh = np.linalg.svd(U_Mat, full_matrices=True)
Modes = U[:,0:50]
q_Mat = np.matmul(Modes.transpose(),U_Mat)

np.save("q_val.npy",q_Mat)

# ------------------------------------------------------------------
# --------Simulate in Modal Coordinate -----------------------------
#-------------------------------------------------------------------
q = q_Mat[:,0].reshape(-1,1)
q_Mat_new = []
for i in range(Tmax):
  if (i == 0):
   print("Reduced i is ====", i)
   time = str(i)
   q_Mat_new.append(q)      
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
   q = linalg.solve(AR, bR)
   U = np.matmul(Modes,q)
   a.setU(U)
   a.setPrevU()
   a.updatephi()
   print("q is ======", q)   
  else:
   print("Reduced i is ====", i)
   time = str(i)
   a.exportU(".",time,"U")
   q_Mat_new.append(q)      
   A = a.get_system_matrix(U)
   A_array = A.toarray()
   phi_transpose_A = np.matmul(Modes.transpose(),A_array)
   AR = np.matmul(phi_transpose_A,Modes)
   b = a.get_rhs(U) 
   bR =  np.matmul(Modes.transpose(),b)
   q = linalg.solve(AR, bR)
   U = np.matmul(Modes,q)
   a.setU(U)
   a.setPrevU()
   a.updatephi()   
   print("q is ======", q) 

q_Mat_new = np.column_stack(q_Mat_new)
#q_diff = (q_Mat_new).reshape(-1,1)




output_data = q_Mat_new.transpose()
max_val = max(output_data.reshape(-1,1))
output_data = output_data
#print("max_val====",max_val)
#output_data = output_data[:,0:No_Modes]
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
        outputs_U = np.matmul(Modes,outputs_numpy.transpose())
        for i in range(Tmax-1):
        #for i in range(1):
          U = outputs_U[:,i].reshape(-1,1)
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
          
          if (i == 0):
            Residual_Physics =  (torch.matmul(AR_tensor,output_[i+1,:].reshape(-1,1)) - bR_tensor)
          else:
            new_tensor = (torch.matmul(AR_tensor,output_[i+1,:].reshape(-1,1)) - bR_tensor)
            Residual_Physics = torch.cat((Residual_Physics,new_tensor), dim = 0)
        
        """
        Tot_Sim = 0 
        for i in range(0,output_.size(0)-1,10):                      
          for j in range(output_.size(1)):
              Tot_Sim = Tot_Sim + 1 
         
        print("Tot_Sim ===", Tot_Sim)
        # new_tensor = torch.zeros(Tot_Sim)
        dR_dU_Dis = torch.zeros(Tot_Sim,output_.size(0)*output_.size(1))


        # Now populate this dR_dU matrix:
        k = 0
        
        for i in range(0,output_.size(0)-1,10):
          for j in range(output_.size(1)): 
            Up = U
            Um = U
            Up[i,j] = U[i,j] + 0.000001
            Um[i,j] = U[i,j] - 0.000001
            a.setU(Up[])
            a.setPrevU()
            a.updatephi()   
            A = a.get_system_matrix(Up)
            A_array = A.toarray()
            b = a.get_rhs(Up) 
            AR_tensor = torch.from_numpy(A_array).float()
            bR_tensor = torch.from_numpy(b).float().reshape(-1,1)
            R_Up = (torch.matmul(AR_tensor[0:400,0:400],output_[i+1,:].reshape(-1,1))

            # ---------------------------------------------------------------------------
            # --------------------------Test --------------------------------------------
            # ---------------------------------------------------------------------------
            
            a.setU(Um)
            a.setPrevU()
            a.updatephi()   
            A = a.get_system_matrix(Um)
            A_array = A.toarray()
            b = a.get_rhs(Um) 
            AR_tensor = torch.from_numpy(A_array).float()
            bR_tensor = torch.from_numpy(b).float().reshape(-1,1)
            R_Um = (torch.matmul(AR_tensor[0:400,0:400],output_[i+1,:].reshape(-1,1))

            # compute the dR/du now ?
            
            dR_dU = (R_Up - R_Um)/(2*0.00001)              
            dR_dU_Dis[i*output_.size(1):(i+1)*output_.size(1),i*output_.size(1) + j] = dR_dU
          dR_dU_Dis[i*output_.size(1):(i+1)*output_.size(1),(i+1)*output_.size(1):(i+2)*output_.size(1)] = AR_tensor
          k = k+1
        """
        return Residual_Physics
    
    conditions = {
        # 't0':Condition(Span({'t': 0}), initial_condition),
        'D': Condition(Span({'t': input_tensor.reshape(-1,1)}), [burger_equation]),
        'E': Condition(input_points = input_tensor,output_points = output_tensor),
    }
