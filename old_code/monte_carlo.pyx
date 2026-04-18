# cython: language_level=3
import numpy as np
cimport numpy as np
from tqdm import tqdm
import parameters as p
import auxiliary_funcs as af
import collisions as col

cpdef tuple singlestep_mc(object v_current, object D_func, object A_func, double dt, double R, double Phi):
    cdef np.ndarray[double, ndim=1] v_current_arr = np.asarray(v_current, dtype=np.float64).ravel()
    cdef np.ndarray[double, ndim=2] D_loc = D_func(v_current_arr)
    cdef np.ndarray[double, ndim=1] A_loc = A_func(v_current_arr)
    cdef np.ndarray[double, ndim=1] std = np.sqrt(2.0 * np.diag(D_loc) * dt)
    cdef np.ndarray[double, ndim=1] dv_D = std * np.random.standard_normal(size=2)
    cdef np.ndarray[double, ndim=1] dv_A = A_loc * dt
    cdef np.ndarray[double, ndim=1] v_new = v_current_arr + dv_D + dv_A
    cdef bint escape = False

    while v_new[0] < 0.0:
        v_new[0] = -v_new[0]

    while v_new[1] < -1.0 or v_new[1] > 1.0:
        if v_new[1] < -1.0:
            v_new[1] = -2.0 - v_new[1]
        elif v_new[1] > 1.0:
            v_new[1] = 2.0 - v_new[1]

    if np.sqrt(1.0 - (1.0 - Phi**2 / v_new[0]**2) / R) - abs(v_new[1]) < 0.0:
        escape = True

    return v_new, escape

cpdef tuple multistep_mc_nopar(object v_initial, int numsteps, object D_func, object A_func, double dt, double R, double Phi):
    cdef np.ndarray[double, ndim=1] v_current = np.asarray(v_initial, dtype=np.float64).ravel().copy()
    cdef np.ndarray[double, ndim=2] v_store = np.empty((numsteps, 2), dtype=np.float64)
    cdef int store_count = 0
    cdef int step
    cdef bint escape = False
    cdef bint trapped = False
    cdef np.ndarray[double, ndim=1] v_new

    for step in range(numsteps):
        v_new, escape = singlestep_mc(v_current, D_func, A_func, dt, R, Phi)
        if escape:
            break
        if v_new[0] <= 0.1:
            trapped = True
            break
        v_store[store_count, 0] = v_new[0]
        v_store[store_count, 1] = v_new[1]
        store_count += 1
        v_current = v_new

    return v_store[:store_count].copy(), escape, trapped

cpdef tuple run_mc_nopar(object source, int numsteps, object D_func, object A_func, double dt, double R, double Phi):
    cdef object source_obj = np.atleast_2d(np.asarray(source, dtype=np.float64))
    cdef int numparticles = source_obj.shape[0]
    cdef int particle
    cdef list loss_velocity = []
    cdef list loss_steps = []
    cdef object trajectory
    cdef bint escape
    cdef bint trapped
    cdef np.ndarray[double, ndim=1] particle_state

    for particle in tqdm(range(numparticles), desc=f"MC particles"):
        particle_state = np.asarray(source_obj[particle], dtype=np.float64).ravel()
        trajectory, escape, trapped = multistep_mc_nopar(particle_state, numsteps, D_func, A_func, dt, R, Phi)
        if trapped:
            continue
        if trajectory.shape[0] == 0:
            continue
        loss_velocity.append(trajectory[-1])
        loss_steps.append(trajectory.shape[0])

    return np.array(loss_velocity), np.array(loss_steps)

cpdef tuple multistep_mc(object v_initial, int numsteps, object D_func, object A_func, double dt, double R, double Phi):
    cdef np.ndarray[double, ndim=1] v_current = np.asarray(v_initial, dtype=np.float64).ravel().copy()
    cdef np.ndarray[double, ndim=2] v_store = np.empty((numsteps, 2), dtype=np.float64)
    cdef int store_count = 0
    cdef int step
    cdef bint escape = False
    cdef np.ndarray[double, ndim=1] v_new

    for step in range(numsteps):
        v_new, escape = singlestep_mc(v_current, D_func, A_func, dt, R, Phi)
        if escape:
            break
        v_store[store_count, 0] = v_new[0]
        v_store[store_count, 1] = v_new[1]
        store_count += 1
        v_current = v_new

    return v_store[:store_count].copy(), escape

cpdef tuple run_mc(object source, int numsteps, object D_func, object A_func, double dt, double R, double Phi):
    cdef object source_obj = np.atleast_2d(np.asarray(source, dtype=np.float64))
    cdef int numparticles = source_obj.shape[0]
    cdef int particle
    cdef list loss_velocity = []
    cdef list loss_steps = []
    cdef object trajectory
    cdef bint escape
    cdef np.ndarray[double, ndim=1] particle_state

    for particle in tqdm(range(numparticles), desc=f"MC particles"):
        particle_state = np.asarray(source_obj[particle], dtype=np.float64).ravel()
        trajectory, escape = multistep_mc(particle_state, numsteps, D_func, A_func, dt, R, Phi)
        if trajectory.shape[0] == 0:
            continue
        loss_velocity.append(trajectory[-1])
        loss_steps.append(trajectory.shape[0])

    return np.array(loss_velocity), np.array(loss_steps)
