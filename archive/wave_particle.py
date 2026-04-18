""" 
Wave particle contributions to the quasilinear diffusion coefficient. 
"""
import numpy as np
import parameters as p

# karney - electrostatic wave perpendicular to magnetic field, zero parallel
def D_karney(v_loc, w, k, E_0): 
    '''
    E_0: electric field amplitude? 
    w: omega, wave frequency
    k: wave vector (k_par, k_perp)
    v: particle velocity (v_par, v_perp)
    '''
    kv_perp = k[1] * v_loc[1]

    # need to account for case - what if v_perp < w/k - no resonance possible? 
    if kv_perp**2 <= w**2: 
        return [0, 0]
    else: 
        return [0, 0.5 * (p.CHARGE_ALPHA * E_0 / p.MASS_ALPHA)**2 * (w / kv_perp) ** 2 / (kv_perp**2 - w**2)**0.5]