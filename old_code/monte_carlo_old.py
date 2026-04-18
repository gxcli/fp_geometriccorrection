
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.random import multivariate_normal

import parameters as p
import auxiliary_funcs as af 
import old_code.collisions_old as col


def singlestep_mc(v_current, D_func, A_func, dt, R, Phi):
    D_loc = D_func(v_current)
    A_loc = A_func(v_current)

    std = np.sqrt(2 * np.diag(D_loc) * dt)
    dv_D = std * np.random.standard_normal(size=len(v_current))
    dv_A = A_loc * dt

    v_new = v_current + dv_D + dv_A
    escape = False 

    while v_new[0] < 0:  # reflect at x=0
        v_new[0] = - v_new[0]

    while v_new[1] < -1 or v_new[1] > 1: # bounce by reflection at xi=\pm 1
        if v_new[1] < -1:
            v_new[1] = -2 - v_new[1]    
        elif v_new[1] > 1:
            v_new[1] = 2 - v_new[1]
    
    lc_condition = np.sqrt(1 - (1 - Phi**2/v_new[0]**2) / R) - np.abs(v_new[1]) # track if in loss cone
    if lc_condition < 0: 
        escape = True

    return v_new, escape



def multistep_mc_nopar(v_initial, numsteps, D_func, A_func, dt, R, Phi): 
    v_current = v_initial.copy()
    v_store = []
    # v_store.append(v_current)

    # for step in tqdm(range(numsteps), desc=f"MC steps x = {v_initial[0]}"):
    for step in range(numsteps):
        v_new, escape = singlestep_mc(v_current, D_func, A_func, dt, R, Phi)
        trapped = False 

        if escape == True: 
            # print('uh oh! escapesies!')
            break 
        elif v_new[0] <= 0.1: # break if reach 0.1
            trapped = True
            break
        else: 
            v_store.append(v_new)
            v_current = v_new

    return np.array(v_store), escape, trapped


def run_mc_nopar(source, numsteps, D_func, A_func, dt, R, Phi): 
    numparticles = source.shape[0]
    loss_velocity = []
    loss_steps = []
    
    for particle in tqdm(range(numparticles), desc=f"MC particles"): 
        trajectory, escape, trapped = multistep_mc_nopar(v_initial=source[particle,:], numsteps=numsteps, D_func=D_func, A_func=A_func, dt=dt, R=R, Phi=Phi)

        if (trapped == True): 
            continue  
        else: 
            if trajectory.size == 0: # if born in loss cone
                continue 

            loss_velocity.append(trajectory[-1])
            loss_steps.append(trajectory.shape[0])

    return np.array(loss_velocity), np.array(loss_steps)


def multistep_mc(v_initial, numsteps, D_func, A_func, dt, R, Phi): 
    v_current = v_initial.copy()
    v_store = []
    # v_store.append(v_current)

    # for step in tqdm(range(numsteps), desc=f"MC steps x = {v_initial[0]}"):
    for step in range(numsteps):
        v_new, escape = singlestep_mc(v_current, D_func, A_func, dt, R, Phi)

        if escape == True: 
            break 
        else: 
            v_store.append(v_new)
            v_current = v_new

    return np.array(v_store), escape


def run_mc(source, numsteps, D_func, A_func, dt, R, Phi): 
    numparticles = source.shape[0]
    loss_velocity = []
    loss_steps = []
    
    for particle in tqdm(range(numparticles), desc=f"MC particles"): 
        trajectory, escape = multistep_mc(v_initial=source[particle,:], numsteps=numsteps, D_func=D_func, A_func=A_func, dt=dt, R=R, Phi=Phi)

        if trajectory.size == 0: # if born in loss cone
            continue 
        else: 
            loss_velocity.append(trajectory[-1])
            loss_steps.append(trajectory.shape[0])

    return np.array(loss_velocity), np.array(loss_steps)