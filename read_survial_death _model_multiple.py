# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 09:21:46 2020

@author: b308
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# create the folder for npy and results
path = "/Yu-Chuan/ForYuChuan/python program/survial_death_model/CLE/multiple/"
f_dir = path + 'result/'

try:
    os.makedirs(f_dir)
except FileExistsError:
    print("The directory has been created on %s" % f_dir)      
except OSError:
    print ("Creation of the directory %s failed" % f_dir)  
else:
    print ("Successfully created the directory %s" % f_dir)

#%% data loading
file_name = 'SDM_CLE_multiple.npy'

data = np.load(path + file_name, allow_pickle=True)

t_sample = data[0]
n_sample = data[1] 

#%% histogram of death time of cell
# thershold is half of initial [caspase3] (ausumed) 
sample_t = []
for n in n_sample:
    n_50000 = np.where(n[24] <= 60000)
    t_50000 = t_sample[n_50000[0][-1]]
    sample_t.append(t_50000)

sample_hr =np.array(sample_t)/3600 # unit hour
bins_hr = np.arange(0,13)
plt.hist(sample_hr, density = True, bins =bins_hr)
plt.xlabel('time (hours)', fontsize = 14)
plt.ylabel('Prob density', fontsize = 14)
file_name = f_dir + 'death_distribution' + '.png' 
plt.savefig(file_name)

#%% all trajectaries of cascase3a 
plt.figure()
for s in n_sample:
    plt.plot(np.array(t_sample)/3600, s[24])
plt.xlabel('time (hours)', fontsize = 14)
plt.ylabel('molecules', fontsize = 14)

#%% satistics analysis
# mean and std
data_mean = np.mean(n_sample, axis = 0)
data_std = np.std(n_sample, axis = 0)

#%% plot each mean trajetories with std
# all varibles
si = ['TNF', 'TNFR1', 'TNFR1a' , 'TRADD', 'TNFR1a_TRADD', 'TRAF2', 'early_complex',                  
    'RIPK1','early_complex_RIPK1', 'IKK', 'early_complex_RIPK1_IKK', 'IKKa',
    'IκB_NFκB', 'IκB_NFκB_IKKa', 'IκBp', 'NFκB', 
    'FADD', 'early_complex_RIPK1_FADD', 'TRADD_TRAF2_RIPK1_FADD', 
    'Caspase8', 'TRADD_TRAF2_RIPK1_FADD_Caspase8', 'Caspase8a', 'Caspase3', 'Caspase8a_Caspase3',
    'Caspase3a', 'DNA_fragmentation', 'cIAP', 'Caspase3a_cIAP' ,'DNA', 'Caspase3a_DNA', 'IκB']

for i, d in enumerate(data_mean):
    plt.figure(figsize=(8.5,6), linewidth = 1.5)
    plt.plot((t_sample/60),d + data_std[i], '#EDBB99')
    plt.plot((t_sample/60),d - data_std[i], '#EDBB99')
    plt.plot((t_sample/60),d)
    plt.xlabel('time (min)', fontsize = 18)
    plt.ylabel('molecular numbers', fontsize = 18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    film_name = f_dir+ si[i] + '.png'
    plt.savefig(film_name, dpi= 1500)

  
#%% NFkB, IKB, NFkB_IkB
plt.figure(figsize=(8.5,6), linewidth = 1.5)
file_name = f_dir + 'NFkB_IkB_comp.png'
plt.plot(t_sample/60, data_mean[12,:]) #NFkB_IkB
plt.plot(t_sample/60, data_mean[15,:]) #NFkB
plt.plot(t_sample/60, data_mean[30,:]) #IkB
plt.legend(['NF-$\kappa$B_I$\kappa$B','NF-$\kappa$B', 'I$\kappa$B'], fontsize = 16)
plt.xlabel('time (min)', fontsize = 18)
plt.ylabel('molecular numbers', fontsize = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(file_name , dpi= 1500)    
 

#%% TNF, NFkB, IKB
plt.figure(figsize=(8.5,6), linewidth = 1.5)
file_name = f_dir + 'TNF_NFkB_IkB.png'
plt.plot(t_sample/60, data_mean[0,:]) #TNF
plt.plot(t_sample/60, data_mean[15,:]) #NFkB
plt.plot(t_sample/60, data_mean[30,:]) #IkB
plt.legend(['TNF','NF-$\kappa$B', 'I$\kappa$B'], fontsize = 16)
plt.xlabel('time (min)', fontsize = 18)
plt.ylabel('molecular numbers', fontsize = 18)
#plt.ylim([0, 500])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(file_name , dpi= 1500)  

#%% survival complex and death complex
file_name = f_dir + 'survival_and_death_complex.png'
plt.figure(figsize=(8.5,6), linewidth = 1.5)
plt.plot(t_sample/60, data_mean[10,:]) #survival
plt.plot(t_sample/60, data_mean[20,:]) #death
plt.legend(['survival complex','death comoplex'], fontsize = 16)
plt.xlabel('time (min)', fontsize = 18)
plt.ylabel('molecular number', fontsize = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig(file_name , dpi= 1500)

#%% Caspase3,  Caspase3_IAP
file_name = f_dir + 'caspase3_Caspase3_IAP.png'
plt.figure(figsize=(8.5,6), linewidth = 1.5)
plt.plot(t_sample/60, data_mean[24,:]) #capase3a
plt.plot(t_sample/60, data_mean[27,:]) #capase3a/cIAP
plt.legend(['capase3a','capase3a_cIAP'], fontsize = 16)
plt.xlabel('time (min)', fontsize = 18)
plt.ylabel('molecular number', fontsize = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 10000)
plt.savefig(file_name , dpi= 1500)

