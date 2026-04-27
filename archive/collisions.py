import numpy as np
import parameters as p
import archive.auxiliary_funcs as af


# PITCH ANGLE ##########################################################
def Asim_xi(v_current, reg=False): 
    x, xi = v_current[0], v_current[1] 
    drift = - 2 * xi / x**3 * (p.ZPERP * af.wperpb_erf(x) - 1 / (4 * x**2) * af.wb_slp(x))

    if reg: 
        reg_drift = - xi / x**2 * (4 * p.ZPERP / np.sqrt(np.pi) * af.WPERPB_MONO - 2 / (3 * np.sqrt(np.pi)) * af.WB_CUBE)
        return np.where(x < 1e-7, reg_drift, drift)
    else:
        return drift


def D_xixi(v_current, reg=True): 
    x, xi = v_current[0], v_current[1] 
    coefficient = 1 / x**3 * (p.ZPERP * af.wperpb_erf(x) - 1 / (4 * x**2) * af.wb_slp(x)) * (1-xi**2)

    if reg: 
        reg_coefficient = 1 / x**2 * (2 * p.ZPERP / np.sqrt(np.pi) * af.WPERPB_MONO - 1 / (3 * np.sqrt(np.pi)) * af.WB_CUBE)
        return np.where(x < 1e-6, reg_coefficient, coefficient)
    else: 
        return coefficient
    

# ENERGY ################################################################
def Asim_x(v_current, reg=True): # REGULARIZE 
    '''
    Asim_x including geometrical correction
    '''
    x = v_current[0]
    A_a = Aa_x(v_current, reg=reg)

    A_geom = - 1 / (2 * x**4) * af.wb_slp(x) + 2 / (x * np.sqrt(np.pi)) * af.wb_cube_exp(x)

    if reg: 
        reg_geom = 4 / (3 * np.sqrt(np.pi) * x) * af.WB_CUBE
        return A_a + np.where(x < 1e-7, reg_geom, A_geom)
    else:
        return A_a + A_geom


def Aa_x(v_current, reg=True):
    '''
    Asim_x without geometrical correction
    '''
    x = v_current[0]
    drift = - p.ZPAR / x**2 * af.wparb_slp(x)
    if reg: 
        reg_drift = - p.ZPAR * 4 / (3 * np.sqrt(np.pi)) * x * af.WPARB_CUBE
        return np.where(x < 1e-10, reg_drift, drift)
    else:
        return drift


def D_xx(v_current, reg=True):
    x = v_current[0]
    coefficient = 1 / (2 * x**3) * af.wb_slp(x)
    if reg:
        return np.where(x < 1e-6, 2 / (3 * np.sqrt(np.pi)) * af.WB_CUBE, coefficient)
    else:
        return coefficient


# TOTAL DRIFT VECTOR AND DIFFUSION TENSOR ###############################
# x, xi means full operator 
# no geom means no geometrical correction
# Mesa means Dxx is zero and no geometrical correction
# should do Mesa where Dxx is zero

def D_xxi(v_current, reg=True): 
    Dxx = D_xx(v_current, reg=reg)
    Dxixi = D_xixi(v_current, reg=reg)
    return np.array([[Dxx,0.0], 
                     [0.0,Dxixi]]) 


def A_xxi(v_current, reg=True): 
    A_x = Asim_x(v_current, reg=reg)
    A_xi = Asim_xi(v_current, reg=reg)
    return np.array([A_x, A_xi])


def A_nogeom(v_current, reg=True): # no geometrical correction
    A_x = Aa_x(v_current, reg=reg)
    A_xi = 0
    return np.array([A_x, A_xi])


def D_nopar(v_current, reg=True): 
    Dxx = 0 
    Dxixi = D_xixi(v_current, reg=reg)
    return np.array([[Dxx,0.0], 
                     [0.0,Dxixi]])


def A_nopar(v_current, reg=True): # Dxx = 0 with geometrical correction 
    A_x = Aa_x(v_current, reg=reg)
    A_xi = Asim_xi(v_current, reg=reg)
    return np.array([A_x, A_xi])


def A_mesa(v_current, reg=True): # Dxx = 0 without geometrical correction
    A_x = Aa_x(v_current, reg=reg)
    A_xi = 0
    return np.array([A_x, A_xi])