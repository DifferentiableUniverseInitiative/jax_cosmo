"""
Created on Jun 12, 2013
@author: Francois Lanusse <francois.lanusse@cea.fr>
"""

doc = ""

doc += "  c : Speed of light in [km/s]\n"
c = 299792.458  # km/s

doc += "  tcmb : Temperature of the CMB today in [K]\n"
tcmb = 2.726  # K

doc += "  rh : Hubble radius in [h^{-1} Mpc]\n"
rh = 2997.92458  # h^{-1} Mpc

doc += "  eta_nu: ratio of energy density in neutrinos to energy in photons\n"
eta_nu = 0.68130

doc += "  h0: Hubble constant in [km/s/(h^{-1} Mpc)]\n"
H0 = 100.0  # km/s/( h^{-1} Mpc)

doc += "  C_1: Instrinsic alignment normalisation constant [(h^2 M_sun Mpc^{-3})^{-1}], see Kirk et al 2010. NB: Bridle & King report different units, but is a typo.\n"
C_1 = 5.0 * 1e-14

doc += "  rho_crit: Critical density of Universe in units of [h^2 M_sun Mpc^{-3}].\n"
rhocrit = 2.7750 * 1e11

__doc__ += "\n".join(sorted(doc.split("\n")))
