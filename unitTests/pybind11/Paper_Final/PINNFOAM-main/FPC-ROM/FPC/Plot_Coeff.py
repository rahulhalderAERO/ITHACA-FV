# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 12:47:29 2024

@author: rahul
"""

import numpy as np
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt




B = np.load("Matrices/B.npy")
C = np.load("Matrices/C.npy")
K = np.load("Matrices/K.npy")
M = np.load("Matrices/M.npy")
P = np.load("Matrices/P.npy")

Red_coeff_005 = scipy.io.loadmat("red_coeff_005/red_coeff.mat")['red_coeff'][:-1,1:]
Red_coeff_006 = scipy.io.loadmat("red_coeff_006/red_coeff.mat")['red_coeff'][:-1,1:]

Red_coeff_a005 = scipy.io.loadmat("red_coeff_005/red_coeff.mat")['red_coeff'][:-1,1:32]
Red_coeff_b005 = scipy.io.loadmat("red_coeff_005/red_coeff.mat")['red_coeff'][:-1,32:]
Red_coeff_a006 = scipy.io.loadmat("red_coeff_006/red_coeff.mat")['red_coeff'][:-1,1:32]
Red_coeff_b006 = scipy.io.loadmat("red_coeff_006/red_coeff.mat")['red_coeff'][:-1,32:]


scaled_pred_dis = scipy.io.loadmat('Output/FPC_Dis_test_0.mat')['predicted_output_test_0']
scaled_pred_data = scipy.io.loadmat('Output/FPC_data.mat')['predicted_output_test_0']



Red_coeff_a = Red_coeff_a005
Red_coeff_b = Red_coeff_b005




dt = 0.01
nu = 0.005

adot = np.zeros((Red_coeff_a.shape[0],Red_coeff_a.shape[1]))
adott_times_C_times_adot = np.zeros((Red_coeff_a.shape[0],Red_coeff_a.shape[1]))

# Start  The Solver:
    
    
for i in range(Red_coeff_a.shape[0]):
    
  for k in range(Red_coeff_a.shape[1]):   

    
    if (i == 0):
      
      adot[i,k] = (1.5*Red_coeff_a[i,k]-2*Red_coeff_a[i,k]+0.5*Red_coeff_a[i,k])/dt
    
    elif (i==1):
     
      adot[i,k] = (1.5*Red_coeff_a[i,k]-2*Red_coeff_a[i-1,k]+0.5*Red_coeff_a[i-1,k])/dt
    
    else:
    
      adot[i,k] = (1.5*Red_coeff_a[i,k]-2*Red_coeff_a[i-1,k]+0.5*Red_coeff_a[i-2,k])/dt
    
      
   
    C_end = C[k,:,:]
    adott_times_C = np.matmul(Red_coeff_a[i,:].reshape(1,-1),C_end)
    adott_times_C_times_adot[i,k] = np.matmul(adott_times_C,Red_coeff_a[i,:].reshape(-1,1))



B_times_a_times_nu = nu*np.matmul(B,Red_coeff_a.transpose()) 
M_times_adot = np.matmul(M,adot.transpose())
K_times_b =  np.matmul(K,Red_coeff_b.transpose())
P_times_a =  np.matmul(P,Red_coeff_a.transpose())


Residual_a =  - M_times_adot + B_times_a_times_nu - adott_times_C_times_adot.transpose() - K_times_b
Residual_b =  P_times_a

# Residual_a[0,:] = 0


t = np.linspace(0,1001,1001)

which_coeff = 3



plt.figure(figsize=(12, 6))


plt.plot(t[0:1001], Red_coeff_a005[0:1001,which_coeff],
          linestyle='--',
          linewidth=2.5,
          color='black',
          label='005')


plt.plot(t[0:1001], scaled_pred_dis[0:1001,which_coeff],
          linestyle='--',
          linewidth=2.5,
          color='red',
          label='005-discretized')

plt.plot(t[0:1001], scaled_pred_data[1001:2002,which_coeff],
          linestyle='--',
          linewidth=2.5,
          color='blue',
          label='005-data')




plt.xlabel('time',fontsize=20)
plt.ylabel('Coeff_{}'.format(which_coeff), fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.xlim(0, 10)
plt.legend(ncol=2, loc=9, fontsize=20) # 9 means top center
plt.tight_layout()
plt.savefig('Output/coeff_prediction_{}.png'.format(which_coeff), dpi=300)
np.save('Output/prediction_Dis.npy',scaled_pred_dis)








  
