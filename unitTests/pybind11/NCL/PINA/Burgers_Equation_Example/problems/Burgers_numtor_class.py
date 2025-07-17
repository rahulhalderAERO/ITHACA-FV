# Call Linear and Nonlinear Matrix

import numpy as np
class Burgers_Discrete_numtor:
     
     def __init__(self,L,T,mu,um1,up1):
     
         self.nx = len(L)
         self.nt = len(T)
         self.mu = mu
         self.L = L
         self.T = T
         self.um1 = um1
         self.up1 = up1
         self.dx = self.L[1]-self.L[0]
         self.dt = self.T[1]-self.T[0] 
         self.const = (self.mu/(self.dx*self.dx))
         
     def Compute_LinearMat(self):
          A =  np.zeros((self.nx, self.nx)) 
          for i in range(1,self.nx-1):
              if (i == 1) :
                  A[i,i+1] = 1
              elif (i == self.nx-2) :
                  A[i,i-1] = 1
              else:
                  A[i,i+1]=1
                  A[i,i-1]=1
              A[i,i] = - 2
          
          return A
      
     def Compute_NonlinearMat(self):
         
         F = np.zeros((self.nx, self.nx))
         for i in range(1,self.nx-1):
             
             if (i == self.nx-2) :
                F[i,i] = -self.um1[i]
             else:
                F[i,i+1]= self.um1[i]
                F[i,i]=  -self.um1[i]
         
         return F