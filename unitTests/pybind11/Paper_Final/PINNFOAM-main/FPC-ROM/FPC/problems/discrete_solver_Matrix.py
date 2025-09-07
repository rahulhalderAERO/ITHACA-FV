# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:41:04 2024

@author: rahul
"""
import numpy as np
import torch

class Residual_Compute:
     
     def __init__(self,dt,mu,Red_coeff_a,Red_coeff_b,B,C,K,M,P):
     
         self.nxa = len(Red_coeff_a)
         self.nxb = len(Red_coeff_b)
         self.mu = mu
         self.dt = dt
         self.Red_coeff_a = Red_coeff_a
         self.Red_coeff_b = Red_coeff_b
         self.B = B
         self.C = C
         self.K = K
         self.M = M
         self.P = P
         
     def Compute_LinearMat(self):
          A =  np.zeros((self.nxa, self.nxa)) 
          for i in range(self.nxa):              
              if (i == 0) :
                  A[i,i] = 0                            
              elif (i == 1) :
                  A[i,i] = 1.5
                  A[i,i-1] = -1.5
              else:
                  A[i,i] = 1.5
                  A[i,i-1] = -2
                  A[i,i-2] = 0.5                  
          A_tensor = (1/self.dt)*torch.from_numpy(A).float()
          return A_tensor
           
     def Adot(self):
         Alin =  self.Compute_LinearMat()
         return torch.sparse.mm(Alin,self.Red_coeff_a)
     
     def B_times_a_times_nu(self):
         return self.mu*torch.matmul(self.B,self.Red_coeff_a.t()) 
     
     def M_times_adot(self):
         return torch.matmul(self.M,self.Adot().t())
     
     def K_times_b(self):
         return torch.matmul(self.K,self.Red_coeff_b.t())
     
     def P_times_a(self):
         return torch.matmul(self.P,self.Red_coeff_a.t())
     
     def at_times_C_times_a(self):
       at_times_C_times_a = torch.zeros((self.Red_coeff_a.shape[0],self.Red_coeff_a.shape[1]))
       for i in range(self.Red_coeff_a.shape[0]):  
         for k in range(self.Red_coeff_a.shape[1]):
            C_end = self.C[k,:,:] 
            at_times_C = torch.matmul(self.Red_coeff_a[i,:].reshape(1,-1),C_end)
            at_times_C_times_a[i,k] = torch.matmul(at_times_C,self.Red_coeff_a[i,:].reshape(-1,1))
       return at_times_C_times_a
    
     def Residual_a(self):         
         return - self.M_times_adot() + self.B_times_a_times_nu() - self.at_times_C_times_a().t() - self.K_times_b()
     
     def Residual_b(self):
         return self.P_times_a()       

            