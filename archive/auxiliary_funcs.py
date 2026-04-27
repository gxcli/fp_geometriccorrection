import numpy as np
import parameters as p 
from scipy import special

# AUXILIARY FUNCTIONS ###########
def slp(x): 
    return -2/np.sqrt(np.pi) * np.exp(-x*x) * x + special.erf(x)


def wparb_slp(x): 
    return (slp(x * p.V_TH_A/p.V_TH_D) * p.C_ad/p.MASS_DEUT 
          + slp(x * p.V_TH_A/p.V_TH_T) * p.C_at/p.MASS_TRIT
          + slp(x * p.V_TH_A/p.V_TH_E) * p.C_ae/p.MASS_ELEC) / p.WPAR_B_DEN


def wb_slp(x):
    return (slp(x * p.V_TH_A/p.V_TH_D) * p.C_ad * p.TEMP_DEUT / p.MASS_DEUT 
          + slp(x * p.V_TH_A/p.V_TH_T) * p.C_at * p.TEMP_TRIT / p.MASS_TRIT
          + slp(x * p.V_TH_A/p.V_TH_E) * p.C_ae * p.TEMP_ELEC / p.MASS_ELEC) / p.WB_DEN


def wb_cube_exp(x):
    return (np.exp(-(x * p.V_TH_A/p.V_TH_D)**2) * (p.V_TH_A/p.V_TH_D)**3 * p.C_ad * p.TEMP_DEUT / p.MASS_DEUT 
          + np.exp(-(x * p.V_TH_A/p.V_TH_T)**2) * (p.V_TH_A/p.V_TH_T)**3 * p.C_at * p.TEMP_TRIT / p.MASS_TRIT
          + np.exp(-(x * p.V_TH_A/p.V_TH_E)**2) * (p.V_TH_A/p.V_TH_E)**3 * p.C_ae * p.TEMP_ELEC / p.MASS_ELEC) / p.WB_DEN


def wperpb_erf(x):
    return (special.erf(x * p.V_TH_A/p.V_TH_D) * p.C_ad
          + special.erf(x * p.V_TH_A/p.V_TH_T) * p.C_at
          + special.erf(x * p.V_TH_A/p.V_TH_E) * p.C_ae) / p.WPERP_B_DEN


WB_CUBE = ((p.V_TH_A/p.V_TH_D)**3 * p.C_ad * p.TEMP_DEUT / p.MASS_DEUT 
        + (p.V_TH_A/p.V_TH_T)**3 * p.C_at * p.TEMP_TRIT / p.MASS_TRIT
        + (p.V_TH_A/p.V_TH_E)**3 * p.C_ae * p.TEMP_ELEC / p.MASS_ELEC) / p.WB_DEN

WPARB_CUBE = ((p.V_TH_A/p.V_TH_D)**3 * p.C_ad / p.MASS_DEUT 
            + (p.V_TH_A/p.V_TH_T)**3 * p.C_at / p.MASS_TRIT
            + (p.V_TH_A/p.V_TH_E)**3 * p.C_ae / p.MASS_ELEC) / p.WPAR_B_DEN

WPERPB_MONO = ((p.V_TH_A/p.V_TH_D) * p.C_ad
             + (p.V_TH_A/p.V_TH_T) * p.C_at 
             + (p.V_TH_A/p.V_TH_E) * p.C_ae) / p.WPERP_B_DEN