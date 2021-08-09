# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:32:42 2020

@author: b308 Yu-Chuan Chen
"""
import numpy as np
import matplotlib.pyplot as plt
import os

#variable name
si = ['TNF', 'TNFR1', 'TNFR1a' , 'TRADD', 'TNFR1a_TRADD', 'TRAF2', 'early_complex',                  
    'RIPK1','early_complex_RIPK1', 'IKK', 'early_complex_RIPK1_IKK', 'IKKa',
    'IκB_NFκB', 'IκB_NFκB_IKKa', 'IκBp', 'NFκB', 
    'FADD', 'early_complex_RIPK1_FADD', 'TRADD_TRAF2_RIPK1_FADD', 
    'Caspase8', 'TRADD_TRAF2_RIPK1_FADD_Caspase8', 'Caspase8a', 'Caspase3', 'Caspase8a_Caspase3',
    'Caspase3a', 'DNA_fragmentation', 'cIAP', 'Caspase3a_cIAP' ,'DNA', 'Caspase3a_DNA', 'IκB']

#%% create data fold
path = 'D:/Yu-Chuan/ForYuChuan/python program/survial_death_model/ODE/molecule_nImpluse/'

def createFold(f_dir): 
    try:
        os.makedirs(f_dir)
    except FileExistsError:
        print("The directory has been created on %s" % f_dir)      
    except OSError:
        print ("Creation of the directory %s failed" % f_dir)  
    else:
        print ("Successfully created the directory %s" % f_dir)
        
# f_dir = path + folder
f_dir = path + 'result/'         
createFold(f_dir)

# npy saving
save_name = path + 'SDM_ODE_molecule_nImpluse'

# impulse in 1.5 hr for 4 times 
time_2 = 3600*1.5* np.array([1, 3, 7, 9]).reshape(4,1)

#%% kinetic Parameter and initial value

# 1nM in 1 pl = 600 molecules
nM2molecule                             = 600.

k1                                      = 0.185 *1e-3 /nM2molecule   # sce-1*nM-1 
k2                                      = 0.00125 *1e-3  
k3                                      = 0.185 *1e-3  /nM2molecule # sce-1*nM-1
k4                                      = 0.00125 *1e-3 
k5                                      = 0.185 *1e-3  /nM2molecule # sce-1*nM-1
k6                                      = 0.00125 *1e-3 
k7                                      = 0.185 *1e-3   /nM2molecule# sce-1*nM-1
k8                                      = 0.00125 *1e-3 
k9                                      = 0.185 *1e-3 /nM2molecule # sce-1*nM-1
k10                                     = 0.00125 *1e-3
k11                                     = 0.37 *1e-3    
k12                                     = 0.014  *1e-3 /nM2molecule# sce-1*nM-1
k13                                     = 0.00125 *1e-3
k14                                     = 0.37  *1e-3                                        
k15                                     = 0.185 *1e-3 /nM2molecule# sce-1*nM-1
k16                                     = 0.00125 *1e-3
k17                                     = 0.37 *1e-3    
k18                                     = 0.5 *1e-3 /nM2molecule # sce-1*nM-1
k19                                     = 0.2 *1e-3
k20                                     = 0.1 *1e-3     
k21                                     = 0.1  *1e-3 /nM2molecule# sce-1*nM-1
k22                                     = 0.06 *1e-3
k23                                     = 100  *1e-3
k24                                     = 0.185 *1e-3 /nM2molecule   # sce-1*nM-1
k25                                     = 0.00125 *1e-3
k26                                     = 0.37 *1e-3
k27                                     = 0.37  *1e-3
k28                                     = 0.5  *1e-3 /nM2molecule   # sce-1*nM-1
k29                                     = 750  *1e-3 /nM2molecule  # sce-1*nM-1
p                                       = 1.75 *1e-3    

#initial value
a = 10.
TNF                                     =  a  *nM2molecule #1
TNFR1                                   = 100. *nM2molecule#2
TNFR1a                                  =  0.  #3 
TRADD                                   = 150. *nM2molecule#4
TNFR1a_TRADD                            =  0.  #5
TRAF2                                   = 100. *nM2molecule#6
early_complex                           = 0.   #7 TNFR1a_TRADD_TRAF2
RIPK1                                   = 100.*nM2molecule #8
early_complex_RIPK1                     = 0.   #9 # early complex
IKK                                     = 100. *nM2molecule#10
early_complex_RIPK1_IKK                 = 0.   #11 # survival complex 
IKKa                                    = 0.   #12
IκB_NFκB                                = 250.*nM2molecule #13
IκB_NFκB_IKKa                           = 0.   #14
IκBp                                    = 0.   #15
NFκB                                    = 0.   #16
FADD                                    = 100. *nM2molecule#17
early_complex_RIPK1_FADD                = 0.   #18
TRADD_TRAF2_RIPK1_FADD                  = 0.  *nM2molecule #19 (compleII)
Caspase8                                = 80. *nM2molecule#20
TRADD_TRAF2_RIPK1_FADD_Caspase8         = 0.   #21 
Caspase8a                               = 0.   #22
Caspase3                                = 200.*nM2molecule#23
Caspase8a_Caspase3                      = 0.   #24
Caspase3a                               = 0.   #25
DNA_fragmentation                       = 0.   #26
cIAP                                    = 0.   #27
Caspase3a_cIAP                          = 0.   #28
DNA                                     = 800.*nM2molecule#29
Caspase3a_DNA                           = 0.   #30
IκB                                     = 0.   #31

#%% functions
def stoichio_M (var):
    rxn = 2*var
    V = np.zeros((var,rxn))
    for i in range(var):
        V[i, 2*i] = 1
        V[i, 2*i+1] = -1    
    return V

def model(P, dt, NFkB_delay):
    
    [TNF, TNFR1, TNFR1a , TRADD, TNFR1a_TRADD, TRAF2, early_complex,                  
    RIPK1,early_complex_RIPK1, IKK, early_complex_RIPK1_IKK, IKKa,
    IκB_NFκB, IκB_NFκB_IKKa, IκBp, NFκB, 
    FADD, early_complex_RIPK1_FADD, TRADD_TRAF2_RIPK1_FADD, 
    Caspase8, TRADD_TRAF2_RIPK1_FADD_Caspase8, Caspase8a, Caspase3, Caspase8a_Caspase3,
    Caspase3a, DNA_fragmentation, cIAP, Caspase3a_cIAP, DNA, Caspase3a_DNA, IκB ] = P   
    NFkB_delay =  NFkB_delay
    
    D = np.array([
            # TNF  c1
            k2*TNFR1a,
            k1*TNF*TNFR1,
            # TNFR1  c2
            (k2*TNFR1a + k17*early_complex_RIPK1_FADD + k11* early_complex_RIPK1_IKK)*10**(-1.7),
            k1*TNF*TNFR1a,
            #TNFR1a c3
            k1*TNF*TNFR1 + k4* TNFR1a_TRADD,
            k2*TNFR1a + k3* TNFR1a * TRADD,
            # TRADD c4
            k4* TNFR1a_TRADD + k11* early_complex_RIPK1_IKK + k20* TRADD_TRAF2_RIPK1_FADD_Caspase8,
            k3* TNFR1a* TRADD,
            # TNFR1a_TRADD c5
            k3* TNFR1a* TRADD + k6* early_complex,
            k4* TNFR1a_TRADD + k5* TNFR1a_TRADD* RIPK1,
            # RIPK1 c6
            k6* early_complex + k11* early_complex_RIPK1_IKK + k20* TRADD_TRAF2_RIPK1_FADD_Caspase8,
            k5* TNFR1a_TRADD * RIPK1,
            # early_complex c7
            k5* TNFR1a_TRADD * RIPK1 + k8* early_complex_RIPK1,
            k6* early_complex + k7* early_complex* RIPK1,
            # RIPK1 c8
            k8* early_complex_RIPK1 + k11* early_complex_RIPK1_IKK+ k20* TRADD_TRAF2_RIPK1_FADD_Caspase8,
            k7* early_complex* RIPK1,
            # early_complex_RIPK1 c9
            k7* early_complex* RIPK1 + k10* early_complex_RIPK1_IKK + k16* early_complex_RIPK1_FADD,
            k8* early_complex_RIPK1 + k9* early_complex_RIPK1*IKK + k15* early_complex_RIPK1 * FADD,
            # IKK c10
            k10* early_complex_RIPK1_IKK + k14* IκB_NFκB_IKKa,
            k9* early_complex_RIPK1* IKK,
            # early_complex_RIPK1_IKK c11
            k9* early_complex_RIPK1* IKK,
            k10* early_complex_RIPK1_IKK + k11* early_complex_RIPK1_IKK,
            # IKKa c12
            k11* early_complex_RIPK1_IKK + k13* IκB_NFκB_IKKa,
            k12* IKKa * IκB_NFκB,
            # IκB_NFκB c13,
            k13* IκB_NFκB_IKKa + k29* NFκB * IκB,
            k12* IKKa * IκB_NFκB,
            # IκB_NFκB_IKKa c14
            k12* IKKa * IκB_NFκB,
            k13* IκB_NFκB_IKKa + k14* IκB_NFκB_IKKa, 
            # IκBp c15
            k14* IκB_NFκB_IKKa,
            0,
            # NFκB c16 
            k14* IκB_NFκB_IKKa,
            k29* NFκB* IκB,
            # FADD c17
            k16* early_complex_RIPK1_FADD + k20* TRADD_TRAF2_RIPK1_FADD_Caspase8,
            k15* early_complex_RIPK1* FADD,
            # early_complex_RIPK1_FADD c18
            k15* early_complex_RIPK1* FADD,
            k16* early_complex_RIPK1_FADD + k17* early_complex_RIPK1_FADD,
            # TRADD_TRAF2_RIPK1_FADD c19
            k17* early_complex_RIPK1_FADD + k19* TRADD_TRAF2_RIPK1_FADD_Caspase8,
            k18* TRADD_TRAF2_RIPK1_FADD* Caspase8,
            # Caspase8 c20
            k19* TRADD_TRAF2_RIPK1_FADD_Caspase8,
            k18* TRADD_TRAF2_RIPK1_FADD* Caspase8,
            # TRADD_TRAF2_RIPK1_FADD_Caspase8 c21
            k18* TRADD_TRAF2_RIPK1_FADD* Caspase8,
            k19* TRADD_TRAF2_RIPK1_FADD_Caspase8 + k20* TRADD_TRAF2_RIPK1_FADD_Caspase8,
            # Caspase8a c22
            k20* TRADD_TRAF2_RIPK1_FADD_Caspase8 + k22* Caspase8a_Caspase3 + k23 * Caspase8a_Caspase3,
            k21* Caspase8a* Caspase3,
            # Caspase3 c23
            k22* Caspase8a_Caspase3 + k26* Caspase3a_DNA,
            k21* Caspase8a* Caspase3,
            # Caspase8a_Caspase3 c24
            k21* Caspase8a* Caspase3,
            k22* Caspase8a_Caspase3 + k23* Caspase8a_Caspase3,
            # Caspase3a c25
            k23* Caspase8a_Caspase3 + k25* Caspase3a_DNA,
            k28*cIAP* Caspase3a + k24* DNA* Caspase3a,
            # DNA_fragmentation c26
            k26* Caspase3a_DNA,
            0,
            # cIAP c27
            p* NFkB_delay,
            k28* cIAP* Caspase3a,
            # Caspase3a_cIAP c28
            k28* cIAP* Caspase3a,
            0,
            # DNA c29
            k25* Caspase3a_DNA,
            k24* Caspase3a* DNA,
            # Caspase3a_DNA c30
            k24* Caspase3a* DNA,
            k25* Caspase3a_DNA + k26* Caspase3a_DNA,
            # IkB c31
            p* NFkB_delay, 
            k29* NFκB* IκB
               
            ]).reshape(2*var,1)
    
    VD = np.matmul(V,D)*dt 
    
    return VD.reshape(len(P))

# Euler method
def euler_claculate(model, P, dt, delay_time, time_2):   
    P[:,0] = x0  
    delay_index = int(delay_time/dt)
    time2_index = np.apply_along_axis(int, 1, time_2/dt)
     
    for i in range(t_step):    
        NFkB_delay = P[15][max(0,i-delay_index)]
        if any(i == time2_index):
                P[0,i] += P[0,0]
        P[:,i+1] = P[:,i] + model(P[:,i], dt ,NFkB_delay)
        
    return P
        
#%% program runnig
# varibles
x0 = np.array([
    TNF, TNFR1, TNFR1a , TRADD, TNFR1a_TRADD, TRAF2, early_complex,                  
    RIPK1,early_complex_RIPK1, IKK, early_complex_RIPK1_IKK, IKKa,
    IκB_NFκB, IκB_NFκB_IKKa, IκBp, NFκB, 
    FADD, early_complex_RIPK1_FADD, TRADD_TRAF2_RIPK1_FADD, 
    Caspase8, TRADD_TRAF2_RIPK1_FADD_Caspase8, Caspase8a, Caspase3, Caspase8a_Caspase3,
    Caspase3a, DNA_fragmentation, cIAP, Caspase3a_cIAP ,DNA, Caspase3a_DNA, IκB
     ])                               

# initial condition
t =  3600*12 
t_step = 100000
t_interval = np.linspace(0, t, t_step)
dt = t_interval[-1]- t_interval[-2]

X = np.zeros((len(x0), t_step+1))
var = len(x0)
X[:,0] = x0 

V = stoichio_M(var)
delay_time = 60*20

# calculate the result
result = euler_claculate(model, X, dt, delay_time, time_2)
np.save(save_name, result)

#%% plot process
XX = result[:, :-1]

for i , x in enumerate(XX):
    plt.figure(figsize=(8.5,6), linewidth = 1.5)
    plt.plot(t_interval/60, x)
    #plt.legend([si[i]])
    plt.xlabel('time (min)', fontsize = 18)
    plt.ylabel('molecules', fontsize = 18) 
    file_name = f_dir  + si[i] + '.png'
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(file_name , dpi= 1500)

