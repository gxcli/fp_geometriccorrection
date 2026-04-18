
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures

import parameters as p
import auxiliary_funcs as af 
import collisions_vec as col


# def _reflect_boundaries(v_new): # reflecting boundaries 
#     v_new = np.asarray(v_new, dtype=float)
#     v_new[..., 0] = np.abs(v_new[..., 0])

#     xi = v_new[..., 1]
#     mask = (xi < -1) | (xi > 1)
#     while np.any(mask):
#         xi = np.where(xi < -1, -2 - xi, np.where(xi > 1, 2 - xi, xi))
#         mask = (xi < -1) | (xi > 1)

#     v_new[..., 1] = xi
#     return v_new


def _reflect_boundaries(v_new):
    v_new = np.asarray(v_new, dtype=float)

    v_new[..., 0] = np.abs(v_new[..., 0])

    xi = v_new[..., 1]
    xi = ((xi + 1) % 4) - 1 # Map to periodic domain of length 4 centered at 0
    xi = np.where(xi > 1, 2 - xi, xi) # Reflect into [-1,1]

    v_new[..., 1] = xi
    return v_new


def singlestep_mc(v_current, D_func, A_func, dt, R, Phi): # single step of multiple particles
    v_current = np.asarray(v_current, dtype=float)
    scalar_input = v_current.ndim == 1
    if scalar_input:
        v_current = v_current.reshape(1, 2)

    D_loc = np.asarray(D_func(v_current), dtype=float)
    A_loc = np.asarray(A_func(v_current), dtype=float)

    if D_loc.ndim == 2:
        D_loc = D_loc[np.newaxis, ...]
    elif D_loc.ndim != 3 or D_loc.shape[-2:] != (2, 2):
        raise TypeError(
            f"D_func must return a (2,2) tensor for a single particle or (N,2,2) for multiple particles, got {D_loc.shape}"
        )

    if A_loc.ndim == 1:
        A_loc = A_loc[np.newaxis, ...]
    elif A_loc.ndim != 2 or A_loc.shape[-1] != 2:
        raise TypeError(
            f"A_func must return a 2-vector for a single particle or an (N,2) array for multiple particles, got {A_loc.shape}"
        )

    std = np.sqrt(2 * np.stack([D_loc[..., 0, 0], D_loc[..., 1, 1]], axis=-1) * dt)
    dv_D = std * np.random.standard_normal(size=v_current.shape)
    dv_A = A_loc * dt

    v_new = v_current + dv_D + dv_A
    v_new = _reflect_boundaries(v_new)

    lc_condition = np.sqrt(1 - (1 - Phi**2 / v_new[..., 0]**2) / R) - np.abs(v_new[..., 1])
    escape = lc_condition < 0

    if scalar_input:
        return v_new[0], bool(escape[0])

    return v_new, escape


def _split_source_chunks(source, nprocs):
    source = np.asarray(source, dtype=float)
    if source.ndim == 1:
        source = source.reshape(1, 2)

    numparticles = source.shape[0]
    if nprocs <= 1 or numparticles == 0:
        return [source]

    nprocs = min(int(nprocs), numparticles)
    chunks = np.array_split(source, nprocs)
    return [chunk for chunk in chunks if chunk.size > 0]


def _run_mc_nopar_single(source, numsteps, D_func, A_func, dt, R, Phi):  # batch of particles
    source = np.asarray(source, dtype=float)
    if source.ndim == 1:
        source = source.reshape(1, 2)

    trap_threshold = Phi # if goes past loss cone, then trapped
    numparticles = source.shape[0]
    v_current = source.copy()
    last_velocity = np.zeros_like(v_current)
    step_counts = np.zeros(numparticles, dtype=int)
    escaped = np.zeros(numparticles, dtype=bool)
    trapped = np.zeros(numparticles, dtype=bool)
    active = np.ones(numparticles, dtype=bool)

    for step in tqdm(range(numsteps)):
        active_idx = np.nonzero(active)[0]
        if active_idx.size == 0:
            break

        v_active = v_current[active_idx]
        v_new_active, escape_active = singlestep_mc(v_active, D_func, A_func, dt, R, Phi)

        step_counts[active_idx] += 1
        trap_active = v_new_active[:, 0] <= trap_threshold

        last_velocity[active_idx] = v_new_active
        escaped[active_idx] = escape_active
        trapped[active_idx] = trap_active

        stop_active = escape_active | trap_active
        continue_active = ~stop_active

        v_current[active_idx[continue_active]] = v_new_active[continue_active]
        active[active_idx[stop_active]] = False

    terminated = escaped
    return last_velocity[terminated], step_counts[terminated], escaped[terminated], trapped[terminated]



def _run_mc_single(source, numsteps, D_func, A_func, dt, R, Phi):
    source = np.asarray(source, dtype=float)
    if source.ndim == 1:
        source = source.reshape(1, 2)

    numparticles = source.shape[0]
    v_current = source.copy()
    last_velocity = np.zeros_like(v_current)
    step_counts = np.zeros(numparticles, dtype=int)
    escaped = np.zeros(numparticles, dtype=bool)
    active = np.ones(numparticles, dtype=bool)

    for step in tqdm(range(numsteps)):
        active_idx = np.nonzero(active)[0]
        if active_idx.size == 0:
            break

        v_active = v_current[active_idx]
        v_new_active, escape_active = singlestep_mc(v_active, D_func, A_func, dt, R, Phi)

        step_counts[active_idx] += 1

        last_velocity[active_idx] = v_new_active
        escaped[active_idx] = escape_active

        stop_active = escape_active
        continue_active = ~stop_active

        v_current[active_idx[continue_active]] = v_new_active[continue_active]
        active[active_idx[stop_active]] = False

    terminated = escaped
    return last_velocity[terminated], step_counts[terminated], escaped[terminated]


def _run_mc_nopar_chunk(args):
    return _run_mc_nopar_single(*args)


def _run_mc_chunk(args):
    return _run_mc_single(*args)


def run_mc_nopar(source, numsteps, D_func, A_func, dt, R, Phi, nprocs=1):
    source = np.asarray(source, dtype=float)
    chunks = _split_source_chunks(source, nprocs)
    if len(chunks) == 1:
        return _run_mc_nopar_single(source, numsteps, D_func, A_func, dt, R, Phi)

    args = [(chunk, numsteps, D_func, A_func, dt, R, Phi) for chunk in chunks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(args)) as executor:
        results = list(executor.map(_run_mc_nopar_chunk, args))

    velocities, steps, escaped, trapped = zip(*results)
    return (
        np.concatenate(velocities, axis=0),
        np.concatenate(steps, axis=0),
        np.concatenate(escaped, axis=0),
        np.concatenate(trapped, axis=0),
    )


def run_mc(source, numsteps, D_func, A_func, dt, R, Phi, nprocs=1):
    source = np.asarray(source, dtype=float)
    chunks = _split_source_chunks(source, nprocs)
    if len(chunks) == 1:
        return _run_mc_single(source, numsteps, D_func, A_func, dt, R, Phi)

    args = [(chunk, numsteps, D_func, A_func, dt, R, Phi) for chunk in chunks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(args)) as executor:
        results = list(executor.map(_run_mc_chunk, args))

    velocities, steps, escaped = zip(*results)
    return (
        np.concatenate(velocities, axis=0),
        np.concatenate(steps, axis=0),
        np.concatenate(escaped, axis=0),
    )