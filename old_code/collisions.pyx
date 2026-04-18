# cython: language_level=3
import numpy as np
cimport numpy as np
import parameters as p
import auxiliary_funcs as af

cpdef double Asim_xi(np.ndarray[double, ndim=1] v_current, bint reg=False):
    cdef double x = v_current[0]
    cdef double xi = v_current[1]
    cdef double drift = -2.0 * xi / (x**3) * (p.ZPERP * af.wperpb_erf(x) - 1.0 / (4.0 * x**2) * af.wb_slp(x))
    cdef double reg_drift

    if reg:
        reg_drift = - xi / (x**2) * (4.0 * p.ZPERP / np.sqrt(np.pi) * af.WPERPB_MONO - 2.0 / (3.0 * np.sqrt(np.pi)) * af.WB_CUBE)
        if x < 1e-7:
            return reg_drift
        return drift
    else:
        return drift

cpdef double D_xixi(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double x = v_current[0]
    cdef double xi = v_current[1]
    cdef double coefficient = 1.0 / (x**3) * (p.ZPERP * af.wperpb_erf(x) - 1.0 / (4.0 * x**2) * af.wb_slp(x)) * (1.0 - xi**2)
    cdef double reg_coefficient

    if reg:
        reg_coefficient = 1.0 / (x**2) * (2.0 * p.ZPERP / np.sqrt(np.pi) * af.WPERPB_MONO - 1.0 / (3.0 * np.sqrt(np.pi)) * af.WB_CUBE)
        if x < 1e-6:
            return reg_coefficient
        return coefficient
    else:
        return coefficient

cpdef double Asim_x(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double x = v_current[0]
    cdef double A_a = Aa_x(v_current, reg=reg)
    cdef double A_geom = -1.0 / (2.0 * x**4) * af.wb_slp(x) + 2.0 / (x * np.sqrt(np.pi)) * af.wb_cube_exp(x)
    cdef double reg_geom

    if reg:
        reg_geom = 4.0 / (3.0 * np.sqrt(np.pi) * x) * af.WB_CUBE
        if x < 1e-7:
            return A_a + reg_geom
        return A_a + A_geom
    else:
        return A_a + A_geom

cpdef double Aa_x(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double x = v_current[0]
    cdef double drift = - p.ZPAR / (x**2) * af.wparb_slp(x)
    cdef double reg_drift
    if reg:
        reg_drift = - p.ZPAR * 4.0 / (3.0 * np.sqrt(np.pi)) * x * af.WPARB_CUBE
        if x < 1e-10:
            return reg_drift
        return drift
    else:
        return drift

cpdef double D_xx(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double x = v_current[0]
    cdef double coefficient = 1.0 / (2.0 * x**3) * af.wb_slp(x)
    if reg:
        if x < 1e-6:
            return 2.0 / (3.0 * np.sqrt(np.pi)) * af.WB_CUBE
        return coefficient
    else:
        return coefficient

cpdef np.ndarray D_xxi(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double Dxx = D_xx(v_current, reg=reg)
    cdef double Dxixi = D_xixi(v_current, reg=reg)
    return np.array([[Dxx, 0.0], [0.0, Dxixi]])

cpdef np.ndarray A_xxi(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double A_x = Asim_x(v_current, reg=reg)
    cdef double A_xi = Asim_xi(v_current, reg=reg)
    return np.array([A_x, A_xi])

cpdef np.ndarray A_nogeom(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double A_x = Aa_x(v_current, reg=reg)
    cdef double A_xi = 0.0
    return np.array([A_x, A_xi])

cpdef np.ndarray D_nopar(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double Dxx = 0.0
    cdef double Dxixi = D_xixi(v_current, reg=reg)
    return np.array([[Dxx, 0.0], [0.0, Dxixi]])

cpdef np.ndarray A_nopar(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double A_x = Aa_x(v_current, reg=reg)
    cdef double A_xi = Asim_xi(v_current, reg=reg)
    return np.array([A_x, A_xi])

cpdef np.ndarray A_alej(np.ndarray[double, ndim=1] v_current, bint reg=True):
    cdef double A_x = Aa_x(v_current, reg=reg)
    cdef double A_xi = 0.0
    return np.array([A_x, A_xi])
