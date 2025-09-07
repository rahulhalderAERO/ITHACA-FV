# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:41:04 2024

@author: rahul
"""
import torch

class Residual_Compute:
     
     def __init__(self,mu,nxa,nxb,i,coeff_c,B,C,K,M,P):
     
         self.nxa = nxa
         self.nxb = nxb
         self.mu = mu
         self.i = i
         self.coeff_c = coeff_c
         self.B = B
         self.C = C
         self.K = K
         self.M = M
         self.P = P
             
     def B_times_a_times_nu(self):
          return self.mu*torch.mm(self.B,self.coeff_c[:self.nxa].reshape(-1,1)) 
     
     def K_times_b(self):
          return torch.mm(self.K,self.coeff_c[self.nxa:].reshape(-1,1))
     
     def P_times_a(self):
          return torch.mm(self.P,self.coeff_c[:self.nxa].reshape(-1,1))
     
     def at_times_C_times_a(self):
        at_times_C_times_a = torch.zeros(self.nxa)         
        for k in range(self.nxa):
             C_end = self.C[k,:,:] 
             at_times_C = torch.mm(self.coeff_c[:self.nxa].reshape(1,-1),C_end)
             at_times_C_times_a[k] = torch.mm(at_times_C,self.coeff_c[:self.nxa].reshape(-1,1))
        return at_times_C_times_a.reshape(-1,1)
        
     def Residual_spatial(self):         
          return self.B_times_a_times_nu() - self.at_times_C_times_a() - self.K_times_b()

     def Residual_b(self):
          return self.P_times_a()            
    
     
     
         

            