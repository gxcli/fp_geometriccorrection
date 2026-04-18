'''
Fokker Planck equation 
with directions with respect to magnetic field 
generate local diffusion coefficient 
'''
import numpy as np
import parameters as p

def slp(x): 
    return 0 

# DIFFUSION #################################### 
def D_fp_parperp(v_loc): 
    v_par, v_perp = v_loc
    v_mag = np.sqrt(v_par*v_par + v_perp*v_perp)

    # perp and par in D_perp, D_par are with respect to the velocity 
    D_perp = p.C_ad * (1 - p.TEMP_DEUT / (p.MASS_DEUT * v_mag**2)) * (p.MASS_ALPHA * v_mag)**(-3) \
            + p.C_at *  (1 - p.TEMP_TRIT / (p.MASS_TRIT * v_mag**2)) * (p.MASS_ALPHA * v_mag)**(-3) \
            + p.C_aa * (1 - p.TEMP_ALPHA / (p.MASS_ALPHA * v_mag**2)) * (p.MASS_ALPHA * v_mag)**(-3)
    
    D_par = p.C_ad * p.TEMP_DEUT / (p.MASS_DEUT * p.MASS_ALPHA**3 * v_mag**5) \
            + p.C_at * p.TEMP_TRIT / (p.MASS_TRIT * p.MASS_ALPHA**3 * v_mag**5) \
            + p.C_aa * p.TEMP_ALPHA / (p.MASS_ALPHA * p.MASS_ALPHA**3 * v_mag**5)
    
    # these directions are with respect to the magnetic field 
    Dparpar = D_perp * v_perp**2 + D_par * v_par**2 
    Dperpperp = D_par * v_par**2 + D_perp * v_perp**2 
    Dparperp = (D_par - D_perp) * v_par * v_perp

    return np.array([[Dparpar,Dparperp],
                     [Dparperp,Dperpperp]])


def D_fp_xxi(v_loc): 
    '''
    v_loc: [x, xi], where x = v/v_tha and xi=v_z/v
    currently assuming only pitch angle scattering and no drag (D_parallel = 0)
    '''
    x, xi = v_loc
    v = x*p.V_TH_A 

    # perp and par in D_perp, D_par are directions with respect to the velocity 
    D_perp = p.C_ad * (1 - p.TEMP_DEUT / (p.MASS_DEUT * v**2)) * (p.MASS_ALPHA * v)**(-3) \
            + p.C_at *  (1 - p.TEMP_TRIT / (p.MASS_TRIT * v**2)) * (p.MASS_ALPHA * v)**(-3)
    
    D_par = p.C_ad * p.TEMP_DEUT / (p.MASS_DEUT * p.MASS_ALPHA**3 * v**5) \
            + p.C_at * p.TEMP_TRIT / (p.MASS_TRIT * p.MASS_ALPHA**3 * v**5)
    
    # Dxx = D_par * x * x # later uncomment, for now, assume only pitch angle scattering 
    Dxx = 0
    Dxixi = D_perp * (1 - xi * xi)

    return np.array([[Dxx,0],
                     [0,Dxixi]])


def D_fp_xxi_norm(v_loc): 
    '''
    v_loc: [x, xi], where x = v/v_tha and xi=v_z/v
    Use normalization from Ian's paper
    Currently do not assume only pitch angle scattering. Cut out the alpha-alpha contributions too. Oops.
    Corrected - regularization near the origin. no 1/4x^2 term. 
    '''
    x, xi = v_loc 

    Dxx = 1 / (2 * x**3)
    Dxixi = 1 / x**3 * p.Z_PERP_AI * (1 - xi**2) 
    
    return np.array([[Dxx,0.0], 
                    [0.0,Dxixi]])

# DRIFT #################################### 
def A_fp_parperp(v_loc): 
    v_par, v_perp = v_loc
    v_mag = np.sqrt(v_par*v_par + v_perp*v_perp)

    A_par = p.C_ad * v_par / (p.MASS_DEUT * p.MASS_ALPHA**2 * v_mag**3)  \
            + p.C_at * v_par / (p.MASS_TRIT * p.MASS_ALPHA**2 * v_mag**3) 
            # + p.C_aa * v_par / (p.MASS_ALPHA * p.MASS_ALPHA**2 * v_mag**3) 
    
    A_perp = p.C_ad * v_perp / (p.MASS_DEUT * p.MASS_ALPHA**2 * v_mag**3)  \
            + p.C_at * v_perp / (p.MASS_TRIT * p.MASS_ALPHA**2 * v_mag**3) 
            # + p.C_aa * v_perp / (p.MASS_ALPHA * p.MASS_ALPHA**2 * v_mag**3) 

    return np.array([A_par, A_perp])


def A_fp_xxi(v_loc): 
    x, xi = v_loc
    v = x*p.V_TH_A

    D_perp = p.C_ad * (1 - p.TEMP_DEUT / (p.MASS_DEUT * v**2)) * (p.MASS_ALPHA * v)**(-3) \
            + p.C_at *  (1 - p.TEMP_TRIT / (p.MASS_TRIT * v**2)) * (p.MASS_ALPHA * v)**(-3)
    
    A_par = p.C_ad / (p.MASS_DEUT * p.V_TH_A**3 * p.MASS_ALPHA**2 * x**2)  \
            + p.C_at / (p.MASS_TRIT * p.V_TH_A**3 * p.MASS_ALPHA**2 * x**2) 

    gradD_x = - p.C_ad * p.TEMP_DEUT / (p.MASS_DEUT * p.MASS_ALPHA**3 * p.V_TH_A**5 * x**4) \
              - p.C_at * p.TEMP_TRIT / (p.MASS_TRIT * p.MASS_ALPHA**3 * p.V_TH_A**5 * x**4)
    
    gradD_xi = -2 * xi * D_perp

    return np.array([A_par - gradD_x, -gradD_xi])


def A_fp_xxi_norm(v_loc): 
    '''
    v_loc: [x, xi], where x = v/v_tha and xi=v_z/v
    Use normalization from Ian's paper
    Currently do not assume only pitch angle scattering. Cut out the alpha-alpha contributions too. Oops.
    '''
    x, xi = v_loc

    A_x = - p.Z_PAR_AI * x - 1.0 / (2.0 * x**4)
    A_xi = - 2.0 * xi * p.Z_PERP_AI / x**3 

    return np.array([A_x, A_xi])