# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 15:32:42 2020

@author: b308
"""
import numpy as np
import multiprocessing as mp
from functools import partial
import time

si = ['TNF', 'TNFR1', 'TNFR1a' , 'TRADD', 'TNFR1a_TRADD', 'TRAF2', 'early_complex',                  
    'RIPK1','early_complex_RIPK1', 'IKK', 'early_complex_RIPK1_IKK', 'IKKa',
    'IκB_NFκB', 'IκB_NFκB_IKKa', 'NFκB', 
    'FADD', 'early_complex_RIPK1_FADD', 'TRADD_TRAF2_RIPK1_FADD', 
    'Caspase8', 'TRADD_TRAF2_RIPK1_FADD_Caspase8', 'Caspase8a', 'Caspase3', 'Caspase8a_Caspase3',
    'Caspase3a', 'cIAP', 'Caspase3a_cIAP' , 'IκB']

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
def stoichoi_M (Z, eq_n, Rxn): 
    V = np.zeros([len(Z), Rxn])
    j = 0
    for i, e in enumerate(eq_n):
        V[i, j: j + sum(e)] = np.array([1]*e[0] + [-1]*e[1])
        j += sum(e)
    return V
  
def fixedNoise (Rxn, swap):
    I1 = np.eye(Rxn)
    I2 = np.copy(I1)
    for s in swap:
        I2[s[0]] =  I1[s[1]]
    return I2

def model(P, dt, NFkB_delay):
    
    [TNF, TNFR1, TNFR1a , TRADD, TNFR1a_TRADD, TRAF2, early_complex,                  
    RIPK1,early_complex_RIPK1, IKK, early_complex_RIPK1_IKK, IKKa,
    IκB_NFκB, IκB_NFκB_IKKa, IκBp, NFκB, 
    FADD, early_complex_RIPK1_FADD, TRADD_TRAF2_RIPK1_FADD, 
    Caspase8, TRADD_TRAF2_RIPK1_FADD_Caspase8, Caspase8a, Caspase3, Caspase8a_Caspase3,
    Caspase3a, DNA_fragmentation, cIAP, Caspase3a_cIAP, DNA, Caspase3a_DNA, IκB ] = P   
    # translation time-delay of IkB & cIAP 
    NFkB_delay =  NFkB_delay
    
     # propensity array
    D = np.array([
            # TNF  c1
            k2*TNFR1a, 
                ## decay
            k1*TNF*TNFR1, 
            # TNFR1  c2
            k2*TNFR1a, 
            k17*early_complex_RIPK1_FADD, 
            k11* early_complex_RIPK1_IKK, 
                ## decay
            k1*TNF*TNFR1a, 
            #TNFR1a c3 
            k1*TNF*TNFR1,
            k4* TNFR1a_TRADD, 
                ## decay
            k2*TNFR1a,
            k3*TNFR1a * TRADD, 
            # TRADD c4
            k4* TNFR1a_TRADD, 
            k11* early_complex_RIPK1_IKK, 
            k20* TRADD_TRAF2_RIPK1_FADD_Caspase8, 
                ## decay
            k3* TNFR1a* TRADD, 
            # TNFR1a_TRADD c5
            k3* TNFR1a* TRADD, 
            k6* early_complex,  
                ## decay
            k4* TNFR1a_TRADD, 
            k5* TNFR1a_TRADD* RIPK1, 
            # TRAF2 c6
            k6* early_complex,
            k11* early_complex_RIPK1_IKK, 
            k20* TRADD_TRAF2_RIPK1_FADD_Caspase8, 
                ## decay
            k5* TNFR1a_TRADD * TRAF2, 
            # early_complex c7
            k5* TNFR1a_TRADD * RIPK1, 
            k8* early_complex_RIPK1,
                ## decay
            k6* early_complex, 
            k7* early_complex* RIPK1,
            # RIPK1 c8
            k8* early_complex_RIPK1, 
            k11* early_complex_RIPK1_IKK, 
            k20* TRADD_TRAF2_RIPK1_FADD_Caspase8,
                ## decay
            k7* early_complex* RIPK1,
            # early_complex_RIPK1 c9
            k7* early_complex* RIPK1, 
            k10* early_complex_RIPK1_IKK,
            k16* early_complex_RIPK1_FADD,
                ## decay
            k8* early_complex_RIPK1,
            k9* early_complex_RIPK1*IKK, 
            k15* early_complex_RIPK1 * FADD,
            # IKK c10
            k10* early_complex_RIPK1_IKK, 
            k14* IκB_NFκB_IKKa,
                ## decay
            k9* early_complex_RIPK1* IKK,
            # early_complex_RIPK1_IKK c11
            k9* early_complex_RIPK1* IKK,
                ## decay
            k10* early_complex_RIPK1_IKK, 
            k11* early_complex_RIPK1_IKK,
            # IKKa c12
            k11* early_complex_RIPK1_IKK/2, 
            k13* IκB_NFκB_IKKa,
                ## decay
            k12* IKKa * IκB_NFκB,
            # IκB_NFκB c13,
            k13* IκB_NFκB_IKKa, 
            k29* NFκB * IκB,
                ## decay
            k12* IKKa * IκB_NFκB,
            # IκB_NFκB_IKKa c14
            k12* IKKa * IκB_NFκB,
                ## decay
            k13* IκB_NFκB_IKKa, 
            k14* IκB_NFκB_IKKa, 
            # IκBp c15
            k14* IκB_NFκB_IKKa,
                ## decay
            0,
            # NFκB c16 
            k14* IκB_NFκB_IKKa,
                ## decay
            k29* NFκB* IκB,
            # FADD c17
            k16* early_complex_RIPK1_FADD, 
            k20* TRADD_TRAF2_RIPK1_FADD_Caspase8,
                ## decay
            k15* early_complex_RIPK1* FADD,
            # early_complex_RIPK1_FADD c18
            k15* early_complex_RIPK1* FADD,
                ## decay
            k16* early_complex_RIPK1_FADD, 
            k17* early_complex_RIPK1_FADD,
            # TRADD_TRAF2_RIPK1_FADD c19
            k17* early_complex_RIPK1_FADD, 
            k19* TRADD_TRAF2_RIPK1_FADD_Caspase8,
                ## decay
            k18* TRADD_TRAF2_RIPK1_FADD* Caspase8,
            # Caspase8 c20
            k19* TRADD_TRAF2_RIPK1_FADD_Caspase8,
                ## decay
            k18* TRADD_TRAF2_RIPK1_FADD* Caspase8,
            # TRADD_TRAF2_RIPK1_FADD_Caspase8 c21
            k18* TRADD_TRAF2_RIPK1_FADD* Caspase8,
                ## decay
            k19* TRADD_TRAF2_RIPK1_FADD_Caspase8, 
            k20* TRADD_TRAF2_RIPK1_FADD_Caspase8,
            # Caspase8a c22
            k20* TRADD_TRAF2_RIPK1_FADD_Caspase8, 
            k22* Caspase8a_Caspase3, 
            k23 * Caspase8a_Caspase3,
                ## decay
            k21* Caspase8a* Caspase3,
            # Caspase3 c23
            k22* Caspase8a_Caspase3, 
            k26* Caspase3a_DNA,
                ## decay
            k21* Caspase8a* Caspase3,
            # Caspase8a_Caspase3 c24
            k21* Caspase8a* Caspase3,
                ## decay
            k22* Caspase8a_Caspase3, 
            k23* Caspase8a_Caspase3,
            # Caspase3a c25
            k23* Caspase8a_Caspase3, 
            k25* Caspase3a_DNA,
                ## decay
            k28*cIAP* Caspase3a, 
            k24* DNA* Caspase3a,
            # DNA_fragmentation c26
            k26* Caspase3a_DNA,
                ## decay
            0,
            # cIAP c27
            p* NFkB_delay,
                ## decay
            k28* cIAP* Caspase3a,
            # Caspase3a_cIAP c28
            k28* cIAP* Caspase3a,
                ## decay
            0,
            # DNA c29
            k25* Caspase3a_DNA,
                ## decay
            k24* Caspase3a* DNA,
            # Caspase3a_DNA c30
            k24* Caspase3a* DNA,
                ## decay
            k25* Caspase3a_DNA, 
            k26* Caspase3a_DNA,
            # IkB c31
            p* NFkB_delay,
                ## decay
            k29* NFκB* IκB    
                 ]).reshape(Rxn,1)
                                                
    sqdt = np.sqrt(dt)
    G = np.sqrt(D)    
    N = np.random.randn(Rxn).reshape(Rxn,1)
    VD = np.matmul(V,D)*dt +np.matmul(V,G*sqdt*N)
    
    return VD.reshape(len(P))
 
def main(model, P, dt, delay_time):   
    P[:,0] = x0  
    delay_index = int(delay_time/dt)
    for i in range(t_step):    
        NFkB_delay = P[15][max(0,i-delay_index)]
        change = np.array([P[:,i] + model(P[:,i], dt ,NFkB_delay)])
        change[np.where(change <= 0)] = 0
        P[:,i+1] = change
    return P

def async_multicore(main, pool_n):
    pool = mp.Pool(processes = pool_n)    # Open multiprocessing pool
    result = [] 
    #do computation
    for i in range(n_step):
        res = pool.apply_async(main, args = (model, X, dt, delay_time,))
        result.append(res)
    pool.close()
    pool.join()
    
    return result 

def select_min(data, t_interval, t_step, slices):
    t_n = np.arange(0, t_step, slices)
    t_new = t_interval[t_n]
    d_new = []
    for d in data:
        dd = d[:, t_n]
        d_new.append(dd)   
    return t_new, d_new  


# initial condition
x0 = np.array([
    TNF, TNFR1, TNFR1a , TRADD, TNFR1a_TRADD, TRAF2, early_complex,                  
    RIPK1,early_complex_RIPK1, IKK, early_complex_RIPK1_IKK, IKKa,
    IκB_NFκB, IκB_NFκB_IKKa, IκBp, NFκB, 
    FADD, early_complex_RIPK1_FADD, TRADD_TRAF2_RIPK1_FADD, 
    Caspase8, TRADD_TRAF2_RIPK1_FADD_Caspase8, Caspase8a, Caspase3, Caspase8a_Caspase3,
    Caspase3a, DNA_fragmentation, cIAP, Caspase3a_cIAP ,DNA, Caspase3a_DNA, IκB
     ])                               


eq_n = np.array([[1,1], [3,1], [2,2], [3,1], [2,2], 
                [3,1], [2,2], [3,1], [3,3], [2,1], 
                [1,2], [2,1], [2,1], [1,2], [1,1],
                [1,1], [2,1], [1,2], [2,1], [1,1],
                [1,2], [3,1], [2,1], [1,2], [2,2],
                [1,1], [1,1], [1,1], [1,1], [1,2], 
                [1,1]])

swap = np.array([[2,0], [5,1], [6,1], [8,0], [13, 9], [10, 7], [11, 4], [14, 9],
        [16, 7], [21, 17], [18, 15], [19, 4], [20, 12], [22, 17], [24, 15],
        [29,25], [26,23], [27, 4], [28, 12], [30, 25], [33, 23], [38, 34], 
        [36,31], [39,34], 
        [40, 31], [41, 4], [42, 4], [45, 43], [48, 47],
        [49, 43], [50, 37], [51, 37], [52, 37], [53, 46], [56, 35], [57, 35],
        [58, 54], [60, 59], [64, 62], [63, 61], [65, 62], [66, 61], [67, 55],
        [68, 55], [74, 71], [72, 69], [75, 71], [76, 69], [77, 70], [78, 70],
        [83, 82], [84, 81], [85, 81], [87, 80], [86, 79], [88, 80], [89, 79],
        [90, 82], [91, 83], [92, 46]
        ])


t =  3600*12 
t_step = 100000
t_interval = np.linspace(0, t, t_step)
dt = t_interval[-1]- t_interval[-2]

X = np.zeros((len(x0), t_step + 1))
var = len(x0)
X[:,0] = x0 
Rxn = np.apply_along_axis(sum, 0 ,np.apply_along_axis(sum, 0 ,eq_n))
V = stoichoi_M(X, eq_n, Rxn)
delay_time = 60*20

if __name__ == "__main__" :  
    name = "SDM_CLE_multiple.npy"
    pool_n = 64
    n_step = 1000
    slices = 100
    print('Opening {0} cpus for simulation...'.format(pool_n))
    print('Preparing parameter sets...')
    print('Preparing simulation space...')
    print('Loading all simulations...')
    start_time = time.time() 
    holder = partial(main, )
    result = async_multicore(holder,pool_n)
    end_time = time.time()
    print('Finished all simulations with {0} sec'.format(end_time - start_time))
    print('Saving data...')
    data = [p.get() for p in result]
    # selected saving in linus system    
    data = select_min(data, t_interval, t_step, slices)    
    np.save(name, data)
    print("Saved successfully!")