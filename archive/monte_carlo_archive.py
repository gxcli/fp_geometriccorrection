'''
Monte-Carlo simulation. local diffusion uniform - Green's function. 
'''

import numpy as np
from numpy.random import multivariate_normal

import toedit.fokker_planck as fp

import parameters as p

########################################
def local_green_sample(v_current, D_loc, dt): 
    '''
    v_current: current velocity
    D: diffusion tensor, 2x2 matrix
    dt: timestep 
    '''
    covariant_matrix = 2 * D_loc * dt
    v_new = multivariate_normal(mean=v_current, cov=covariant_matrix, check_valid='warn')
    dv = v_new - v_current
    return dv


def singlestep_mc(v_current, Ds_func, As_func, dt): 
    ''' 
    v_current: current velocity
    Ds_func: list of functions dependent only on v_current, each used to calculate local diffusion tensor
    As_func: list of functions dependent only on v_current, each used to calculate drift vector 
    dt: timestep 

    # given a wave - fourier transform into components? local green's for each mode?
    '''
    Ds_loc = [D_f(v_current) for D_f in Ds_func]

    dvs_diff = []
    for D_loc in Ds_loc:
        dv = local_green_sample(v_current, D_loc, dt)
        dvs_diff.append(dv)

    As_loc = [A_f(v_current) for A_f in As_func]
    dvs_adv = np.array(As_loc) * dt # negative sign included in As_loc itself

    v_new = v_current + np.sum(dvs_diff, axis=0) + np.sum(dvs_adv, axis=0)

    # boundaries 
    # in order to have positive semidefinite D, reflect off X_MIN. also reflect for xi bewteen -1 and 1. 
    while v_new[0] < p.X_MIN_IAN: 
        v_new[0] = 2 * p.X_MIN_IAN - v_new[0]

    while v_new[1] < -1 or v_new[1] > 1: # bounce by reflection
        if v_new[1] < -1:
            v_new[1] = -2 - v_new[1]    
        elif v_new[1] > 1:
            v_new[1] = 2 - v_new[1]

    return v_new


# choose small enough dt!!! 
def multistep_mc(v_initial, numsteps, processes_dict, dt, R, phi): 
    '''
    Kicking around a single particle

    v_initial: initial velocity 
    numsteps: number of steps 
    processes: dictionary, {type:'name' string, w: scalar, k: vector}
    dt: timestep amount
    R: mirror ratio, Bend/Bmid
    phi: confining potential 
    '''
    # input some initial state
    v_current = v_initial.copy()
    v_store = []
    v_store.append(v_current)

    # read in values, later have to move this down to the multirun part - so not calculating multiple times.  
    # D func, A func list of lambda functions, dependent only on v_current 
    Ds_func = []
    As_func = []

    for key, process in processes_dict.items(): 
        if process['type'] == "fokkerplanck xxi norm": 
            Ds_func.append(fp.D_fp_xxi_norm)
            As_func.append(fp.A_fp_xxi_norm)
        else: 
            print(f'{key} unclassified')

    # pass in D func where lambda, only dependent on v_current, D_list 

    for i in range(numsteps): 
        v_new = singlestep_mc(v_current, Ds_func, As_func, dt)
        v_store.append(v_new)
        v_current = v_new

    return np.array(v_store)


