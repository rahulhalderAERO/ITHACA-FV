import numpy as np
import torch
from scipy.io import loadmat


DEIM_sensor = loadmat('Burgers_DEIM_phi.mat')['DEIM_sensor'].T
DEIM_sensor = torch.from_numpy(DEIM_sensor).float()
Modes = loadmat('Burgers_DEIM_phi.mat')['Modes']
Modes = torch.from_numpy(Modes).float()
phi = loadmat('Burgers_DEIM_phi.mat')['phi']
phi = torch.from_numpy(phi).float()
Ar = loadmat('Burgers_DEIM_phi.mat')['Ar']
Ar = torch.from_numpy(Ar).float()
u_Mat = loadmat("Burgers_FOM.mat")['u_Mat']
u_Mat = torch.from_numpy(u_Mat).float()
q_Mat = loadmat("q_Mat.mat")['q_Mat']
q_Mat = torch.from_numpy(q_Mat).float()

nx = 20
nt = 100
mu = 0.05
L = torch.linspace(0, 1, nx).reshape(-1, 1)
T = torch.linspace(0, 0.5, nt).reshape(-1, 1)


class Burgers_Discrete:
     
     def __init__(self,L,T,mu,qm1,qp1,DEIM_sensor,Modes,phi,Ar):
     
         self.nx = len(L)
         self.nt = len(T)
         self.mu = mu
         self.L = L
         self.T = T
         self.qm1 = qm1
         self.qp1 = qp1
         self.dx = self.L[1]-self.L[0]
         self.dt = self.T[1]-self.T[0] 
         self.const = (self.mu/(self.dx*self.dx))
         self.DEIM_sensor = DEIM_sensor
         self.Modes = Modes
         self.phi = phi
         self.Ar = Ar
      
     def Compute_NonlinearMat(self):
         um1 = torch.matmul(self.phi,self.qm1)
         F = torch.zeros((self.nx, self.nx))
         F_reduced = torch.zeros((self.DEIM_sensor.size(0), self.nx))
         for i in range(1,self.nx-1):
             
             if (i == self.nx-2) :
                F[i,i] = -um1[i]
             else:
                F[i,i+1]= um1[i]
                F[i,i]=  -um1[i]
         no_list = self.DEIM_sensor.size(0)
         for i in range(no_list):             
             F_reduced[i,:] = F[int(self.DEIM_sensor[i,0]),:]
         return F_reduced

     def Burgers_Residual_Temp(self):
         return (1/self.dt)*(self.qp1-self.qm1)
     
     def Burgers_Residual_Spatial(self):
         um1 = torch.matmul(self.phi,self.qm1)
         Fnonlin = self.Compute_NonlinearMat()
         Nonlinear_Term_Reduced = torch.matmul(Fnonlin,um1)
         Nonlinear_Term = torch.matmul(self.Modes, Nonlinear_Term_Reduced) 
         F_projected_q =  torch.matmul(self.phi.T,Nonlinear_Term)
         return  - self.const*torch.matmul(self.Ar, self.qm1)+(1.0/self.dx)*F_projected_q
         
      
     def Burgers_Residual(self):
         return (self.Burgers_Residual_Temp() + self.Burgers_Residual_Spatial())
     
        
     
Burgers_Eqn =  Burgers_Discrete(L, T, mu, q_Mat[:,50], q_Mat[:,51], DEIM_sensor, Modes, phi, Ar)
Burgers_Residual = Burgers_Eqn.Burgers_Residual().numpy()
         
          
        
