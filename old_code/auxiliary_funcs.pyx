# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport exp, erf, sqrt
import parameters as p

cpdef double slp(double x):
    return -2.0 / sqrt(np.pi) * exp(-x * x) * x + erf(x)

cpdef double wparb_slp(double x):
    cdef double xd = x * p.V_TH_A / p.V_TH_D
    cdef double xt = x * p.V_TH_A / p.V_TH_T
    cdef double xe = x * p.V_TH_A / p.V_TH_E
    return (slp(xd) * p.C_ad / p.MASS_DEUT
          + slp(xt) * p.C_at / p.MASS_TRIT
          + slp(xe) * p.C_ae / p.MASS_ELEC) / p.WPAR_B_DEN

cpdef double wb_slp(double x):
    cdef double xd = x * p.V_TH_A / p.V_TH_D
    cdef double xt = x * p.V_TH_A / p.V_TH_T
    cdef double xe = x * p.V_TH_A / p.V_TH_E
    return (slp(xd) * p.C_ad * p.TEMP_DEUT / p.MASS_DEUT
          + slp(xt) * p.C_at * p.TEMP_TRIT / p.MASS_TRIT
          + slp(xe) * p.C_ae * p.TEMP_ELEC / p.MASS_ELEC) / p.WB_DEN

cpdef double wb_cube_exp(double x):
    cdef double xd = x * p.V_TH_A / p.V_TH_D
    cdef double xt = x * p.V_TH_A / p.V_TH_T
    cdef double xe = x * p.V_TH_A / p.V_TH_E
    return (exp(-xd * xd) * (p.V_TH_A / p.V_TH_D)**3 * p.C_ad * p.TEMP_DEUT / p.MASS_DEUT
          + exp(-xt * xt) * (p.V_TH_A / p.V_TH_T)**3 * p.C_at * p.TEMP_TRIT / p.MASS_TRIT
          + exp(-xe * xe) * (p.V_TH_A / p.V_TH_E)**3 * p.C_ae * p.TEMP_ELEC / p.MASS_ELEC) / p.WB_DEN

cpdef double wperpb_erf(double x):
    cdef double xd = x * p.V_TH_A / p.V_TH_D
    cdef double xt = x * p.V_TH_A / p.V_TH_T
    cdef double xe = x * p.V_TH_A / p.V_TH_E
    return (erf(xd) * p.C_ad
          + erf(xt) * p.C_at
          + erf(xe) * p.C_ae) / p.WPERP_B_DEN

WB_CUBE = ((p.V_TH_A / p.V_TH_D)**3 * p.C_ad * p.TEMP_DEUT / p.MASS_DEUT
        + (p.V_TH_A / p.V_TH_T)**3 * p.C_at * p.TEMP_TRIT / p.MASS_TRIT
        + (p.V_TH_A / p.V_TH_E)**3 * p.C_ae * p.TEMP_ELEC / p.MASS_ELEC) / p.WB_DEN

WPARB_CUBE = ((p.V_TH_A / p.V_TH_D)**3 * p.C_ad / p.MASS_DEUT
            + (p.V_TH_A / p.V_TH_T)**3 * p.C_at / p.MASS_TRIT
            + (p.V_TH_A / p.V_TH_E)**3 * p.C_ae / p.MASS_ELEC) / p.WPAR_B_DEN

WPERPB_MONO = ((p.V_TH_A / p.V_TH_D) * p.C_ad
             + (p.V_TH_A / p.V_TH_T) * p.C_at
             + (p.V_TH_A / p.V_TH_E) * p.C_ae) / p.WPERP_B_DEN
