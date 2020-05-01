import jax.numpy as np
from . import constants as const

def z2a(z):
    """ converts from redshift to scale factor """
    return 1.0/(1.0 + z)


def a2z(a):
    """ converts from scale factor to  redshift """
    return 1.0/a - 1.0


class cosmology(object):
    r"""Stores all cosmological parameters

    Parameters
    ----------
    h : float
        Reduced hubble constant (def: 0.7)

    Omega_b : float
        Baryonic matter density (def: 0.045)

    Omega_m : float
        Matter density (def: 0.25)

    Omega_de : float
        Dark energy density (def: 0.75)

    w0, wa : float
        Dark energy equation of state parameters
        (def: w0 = -0.95, wa = 0.0)

    n : float
        Scalar spectral index (def: 1.0)

    tau : float
        Reionization optical depth (def: 0.09)

    sigma8 : float
        Fluctuation amplitude at 8 Mpc/h (def: 0.8)

    References
    ----------

    .. bibliography:: biblio.bib
        :all:
    """

    def __init__(self, **kwargs):
        # initialize default cosmology parameters
        self._h = 0.7
        self._Omega_b = 0.045
        self._Omega_m = 0.25
        self._Omega_de = 0.75
        self._w0 = -0.95
        self._wa = 0.0
        self._n = 1.
        self._tau = 0.09
        self._sigma8 = 0.8

        # initialize computation parameters
        self._amin = 0.001    # minimum scale factor
        self._amax = 1.0      # maximum scale factor
        self._na = 512        # number of points in interpolation arrays

        self._kmin = 0.0001     # minimum scale in power spectrum, in Mpc/h
        self._kmax = 1000.0     # maximum scale, in Mpc/h

        # Uses the input keywords to update the cosmology
        self.update(**kwargs)

    def update(self, **kwargs):
        r"""Updates the current cosmology based on the parameters specified
        in input.

        Parameters
        ----------
        h : float
            Reduced hubble constant

        Omega_b : float
            Baryonic matter density

        Omega_m : float
            Matter density

        Omega_de : float
            Dark energy density

        w0, wa : float
            Dark energy equation of state parameters

        n : float
            Scalar spectral index

        tau : float
            Reionization optical depth

        sigma8: float
            Fluctuation amplitude at 8 Mpc/h
        """

        # Look for keywords and update cosmological parameters
        for kw in kwargs:
            if kw == 'h':
                self._h = kwargs[kw]
            elif kw == 'Omega_b':
                self._Omega_b = kwargs[kw]
            elif kw == 'Omega_m':
                self._Omega_m = kwargs[kw]
            elif kw == 'Omega_de':
                self._Omega_de = kwargs[kw]
            elif kw == 'w0':
                self._w0 = kwargs[kw]
            elif kw == 'wa':
                self._wa = kwargs[kw]
            elif kw == 'n':
                self._n = kwargs[kw]
            elif kw == 'tau':
                self._tau = kwargs[kw]
            elif kw == 'sigma8':
                self._sigma8 = kwargs[kw]

        # Check if the makeFlat keyword was used
        if 'makeFlat' in kwargs:
            self._Omega_de = 1.0 - self._Omega_m

        # Setup constant attributes
        self._Omega_dm = self._Omega_m-self._Omega_b    # Dark matter density
        self._Omega = self._Omega_m+self._Omega_de      # Total density
        self._Omega_k = 1.0 - self._Omega               # Curvature

        # Sugiyama (1995, APJS, 100, 281)
        self._gamma = self._Omega_m*self._h * \
            np.exp(-self._Omega_b*(1. + np.sqrt(2.*self._h)/self._Omega_m))

        if self._Omega > 1.0:   # Closed universe
            self._k = 1.0
            self._sqrtk = np.sqrt(np.abs(self._Omega_k))
        elif self._Omega == 1.0:  # Flat universe
            self._k = 0
            self._sqrtk = 1.
        elif self._Omega < 1.0:  # Open Universe
            self._k = -1.0
            self._sqrtk = np.sqrt(np.abs(self._Omega_k))

        #############################################
        # Quantities computed from 1998:EisensteinHu
        # Provides : - k_eq   : scale of the particle horizon at equality epoch
        #            - z_eq   : redshift of equality epoch
        #            - R_eq   : ratio of the baryon to photon momentum density
        #                       at z_eq
        #            - z_d    : redshift of drag epoch
        #            - R_d    : ratio of the baryon to photon momentum density
        #                       at z_d
        #            - sh_d   : sound horizon at drag epoch
        #            - k_silk : Silk damping scale
        T_2_7_sqr = (const.tcmb/2.7)**2
        h2 = self.h**2
        w_m = self.Omega_m*h2
        w_b = self.Omega_b*h2

        self._k_eq = 7.46e-2*w_m/T_2_7_sqr / self.h     # Eq. (3) [h/Mpc]
        self._z_eq = 2.50e4*w_m/(T_2_7_sqr)**2          # Eq. (2)

        # z drag from Eq. (4)
        b1 = 0.313*np.power(w_m, -0.419)*(1.0+0.607*np.power(w_m, 0.674))
        b2 = 0.238*np.power(w_m, 0.223)
        self._z_d = 1291.0*np.power(w_m, 0.251)/(1.0+0.659*np.power(w_m, 0.828)) * \
            (1.0 + b1*np.power(w_b, b2))

        # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
        self._R_d = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_d)
        # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
        self._R_eq = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_eq)
        # Sound horizon at drag epoch in h^-1 Mpc Eq. (6)
        self._sh_d = 2.0/(3.0*self._k_eq) * np.sqrt(6.0/self._R_eq) * \
            np.log((np.sqrt(1.0 + self._R_d) + np.sqrt(self._R_eq + self._R_d)) /
                (1.0 + np.sqrt(self._R_eq)))
        # Eq. (7) but in [hMpc^{-1}]
        self._k_silk = 1.6 * np.power(w_b, 0.52) * np.power(w_m, 0.73) * \
            (1.0 + np.power(10.4*w_m, -0.95)) / self.h
        #############################################

        #############################################
        # Quantities computed from 1999:EfstathiouBond
        # Provides : - z_r   : redshift of recombination
        #            - sh_r  : sound horizon at recombination
        # z recombination from Eq. (20)
        g1 = 0.078*w_b**(-0.238) * (1.0 + 39.5*w_b**0.7630)**(-1)
        g2 = 0.56*(1.0 + 21.1*w_b**1.81)**(-1)
        a_r = 1.0/(1048.*(1.0+0.00124*w_b**(-0.738))*(1.0+g1*w_m**g2) + 1)
        self._z_r = a2z(a_r)

        # sound horizon at recombination from Eq. (18) and (19)
        a_eq = 1.0/(24185.0 * (1.6813/(1.0 + const.eta_nu)) * w_m)  # Eq. (18)
        R_eq = 30496.*w_b*a_eq  # Eq. (18)
        R_zr = 30496.*w_b*a_r   # Eq. (18)
        frac = (np.sqrt(1.0+R_zr)+np.sqrt(R_zr+R_eq))/(1.0+np.sqrt(R_eq))    # Eq. (19)
        self._sh_r = 4000.0/np.sqrt(w_b)*np.sqrt(a_eq)/np.sqrt(1.0+const.eta_nu) * \
            np.log(frac) * self.h  # Eq. (19) converted to Mpc/h
        #############################################

    def __str__(self):
        return 'FLRW Cosmology with the following parameters: \n' + \
            '    h:        ' + str(self.h) + ' \n' + \
            '    Omega_b:  ' + str(self.Omega_b) + ' \n' + \
            '    Omega_m:  ' + str(self.Omega_m) + ' \n' + \
            '    Omega_de: ' + str(self.Omega_de) + ' \n' + \
            '    w0:       ' + str(self.w0) + ' \n' + \
            '    wa:       ' + str(self.wa) + ' \n' + \
            '    n:        ' + str(self.n) + ' \n' + \
            '    tau:      ' + str(self.tau) + ' \n' + \
            '    sigma8:   ' + str(self.sigma8)

    def w(self, a):
        r"""Dark Energy equation of state parameter using the Linder
        parametrisation.

        Parameters
        ----------
        a : array_like
            Scale factor

        Returns
        -------
        w : ndarray, or float if input scalar
            The Dark Energy equation of state parameter at the specified
            scale factor

        Notes
        -----

        The Linder parametrization :cite:`2003:Linder` for the Dark Energy
        equation of state :math:`p = w \rho` is given by:

        .. math::

            w(a) = w_0 + w (1 -a)
        """
        return self.w0 + (1.0 - a) * self.wa  # Equation (6) in Linder (2003)

    def f_de(self, a):
        r"""Evolution parameter for the Dark Energy density.

        Parameters
        ----------
        a : array_like
            Scale factor

        Returns
        -------
        f : ndarray, or float if input scalar
            The evolution parameter of the Dark Energy density as a function
            of scale factor

        Notes
        -----

        For a given parametrisation of the Dark Energy equation of state,
        the scaling of the Dark Energy density with time can be written as:

        .. math::

            \rho_{de}(a) \propto a^{f(a)}

        (see :cite:`2005:Percival`) where :math:`f(a)` is computed as
        :math:`f(a) = \frac{-3}{\ln(a)} \int_0^{\ln(a)} [1 + w(a^\prime)]
        d \ln(a^\prime)`. In the case of Linder's parametrisation for the
        dark energy in Eq. :eq:`linderParam` :math:`f(a)` becomes:

        .. math::

            f(a) = -3(1 + w_0) + 3 w \left[ \frac{a - 1}{ \ln(a) } - 1 \right]
        """
        # Just to make sure we are not diving by 0
        epsilon = 0.000000001
        return -3.0*(1.0+self.w0) + 3.0*self.wa*((a-1.0)/np.log(a-epsilon) - 1.0)

    def Esqr(self, a):
        r"""Square of the scale factor dependent factor E(a) in the Hubble
        parameter.

        Parameters
        ----------
        a : array_like
            Scale factor

        Returns
        -------
        E^2 : ndarray, or float if input scalar
            Square of the scaling of the Hubble constant as a function of
            scale factor

        Notes
        -----

        The Hubble parameter at scale factor `a` is given by
        :math:`H^2(a) = E^2(a) H_o^2` where :math:`E^2` is obtained through
        Friedman's Equation (see :cite:`2005:Percival`) :

        .. math::

            E^2(a) = \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} a^{f(a)}

        where :math:`f(a)` is the Dark Energy evolution parameter computed
        by :py:meth:`.f_de`.
        """
        return self.Omega_m*np.power(a, -3) + self.Omega_k*np.power(a, -2) + \
            self.Omega_de*np.power(a, self.f_de(a))

    def H(self, a):
        r"""Hubble parameter [km/s/(Mpc/h)] at scale factor `a`

        Parameters
        ----------
        a : array_like
            Scale factor

        Returns
        -------
        H : ndarray, or float if input scalar
            Hubble parameter at the requested scale factor.
        """
        return const.H0 * np.sqrt(self.Esqr(a))

    def Omega_m_a(self, a):
        r"""Matter density at scale factor `a`.

        Parameters
        ----------
        a : array_like
            Scale factor

        Returns
        -------
        Omega_m : ndarray, or float if input scalar
            Non-relativistic matter density at the requested scale factor

        Notes
        -----
        The evolution of matter density :math:`\Omega_m(a)` is given by:

        .. math::

            \Omega_m(a) = \frac{\Omega_m a^{-3}}{E^2(a)}

        see :cite:`2005:Percival` Eq. (6)
        """
        return self.Omega_m * np.power(a, -3) / self.Esqr(a)

    def Omega_de_a(self, a):
        r"""Dark Energy density at scale factor `a`.

        Parameters
        ----------
        a : array_like
            Scale factor

        Returns
        -------
        Omega_de : ndarray, or float if input scalar
            Dark Energy density at the requested scale factor

        Notes
        -----
        The evolution of Dark Energy density :math:`\Omega_{de}(a)` is given
        by:

        .. math::

            \Omega_{de}(a) = \frac{\Omega_{de} a^{f(a)}}{E^2(a)}

        where :math:`f(a)` is the Dark Energy evolution parameter computed by
        :py:meth:`.f_de` (see :cite:`2005:Percival` Eq. (6)).
        """
        return self.Omega_de*np.power(a, self.f_de(a))/self.Esqr(a)

    # def chi2a(self, chi):
    #     r"""Scale factor for the radial comoving distance specified in [Mpc/h].
    #
    #     Parameters
    #     ----------
    #     chi : array_like
    #         Radial comoving distance in [Mpc/h]
    #
    #     Returns
    #     -------
    #     a : ndarray, or float if input scalar
    #         Scale factor corresponding to the specified radial comoving
    #         distance.
    #     """
    #     if self._a_chi_interp is None:
    #         self.a2chi(1.0)
    #
    #     return self._a_chi_interp(chi)

    # def a2chi(self, a):
    #     r"""Radial comoving distance in [Mpc/h] for a given scale factor.
    #
    #     Parameters
    #     ----------
    #     a : array_like
    #         Scale factor
    #
    #     Returns
    #     -------
    #     chi : ndarray, or float if input scalar
    #         Radial comoving distance corresponding to the specified scale
    #         factor.
    #
    #     Notes
    #     -----
    #     The radial comoving distance is computed by performing the following
    #     integration:
    #
    #     .. math::
    #
    #         \chi(a) =  R_H \int_a^1 \frac{da^\prime}{{a^\prime}^2 E(a^\prime)}
    #     """
    #     def dchioverdlna(x):
    #         xa = exp(x)
    #         return self.dchioverda(xa) * xa
    #
    #     chi = vectorize(lambda x: romberg(dchioverdlna, log(x), 0,
    #                                       vec_func=True))
    #
    #     # Initialize interpolation array
    #     if self._chi_a_interp is None:
    #         chitab = chi(self.atab)
    #         self._chi_a_interp = interp1d(self.atab, chitab, kind='quadratic')
    #         self._a_chi_interp = interp1d(chitab[::-1], self.atab[::-1],
    #                                       kind='quadratic')
    #
    #     # For values within the interpolation array use _chi_interp,
    #     # otherwise perform the integration
    #     res = vectorize(lambda x: self._chi_a_interp(x)
    #                     if ((x > self._amin) and (x < self._amax))
    #                     else chi(x))
    #     return res(a)
    #
    # def f_k(self, a):
    #     r"""Transverse comoving distance in [Mpc/h] for a given scale factor.
    #
    #     Parameters
    #     ----------
    #     a : array_like
    #         Scale factor
    #
    #     Returns
    #     -------
    #     f_k : ndarray, or float if input scalar
    #         Transverse comoving distance corresponding to the specified
    #         scale factor.
    #
    #     Notes
    #     -----
    #     The transverse comoving distance depends on the curvature of the
    #     universe and is related to the radial comoving distance through:
    #
    #     .. math::
    #
    #         f_k(a) = \left\lbrace
    #         \begin{matrix}
    #         R_H \frac{1}{\sqrt{\Omega_k}}\sinh(\sqrt{|\Omega_k|}\chi(a)R_H)&
    #             \mbox{for }\Omega_k > 0 \\
    #         \chi(a)&
    #             \mbox{for } \Omega_k = 0 \\
    #         R_H \frac{1}{\sqrt{\Omega_k}} \sin(\sqrt{|\Omega_k|}\chi(a)R_H)&
    #             \mbox{for } \Omega_k < 0
    #         \end{matrix}
    #         \right.
    #     """
    #
    #     chi = self.a2chi(a)
    #     if self.k < 0:      # Open universe
    #         return const.rh/self.sqrtk*sinh(self.sqrtk * chi/const.rh)
    #     elif self.k > 0:    # Closed Universe
    #         return const.rh/self.sqrtk*sin(self.sqrtk * chi/const.rh)
    #     else:
    #         return chi

    # def d_A(self, a):
    #     r"""Angular diameter distance in [Mpc/h] for a given scale factor.
    #
    #     Parameters
    #     ----------
    #     a : array_like
    #         Scale factor
    #
    #     Returns
    #     -------
    #     d_A : ndarray, or float if input scalar
    #
    #     Notes
    #     -----
    #     Angular diameter distance is expressed in terms of the transverse
    #     comoving distance as:
    #
    #     .. math::
    #
    #         d_A(a) = a f_k(a)
    #     """
    #     return a * self.f_k(a)

    def dchioverda(self, a):
        r"""Derivative of the radial comoving distance with respect to the
        scale factor.

        Parameters
        ----------
        a : array_like
            Scale factor

        Returns
        -------
        dchi/da :  ndarray, or float if input scalar
            Derivative of the radial comoving distance with respect to the
            scale factor at the specified scale factor.

        Notes
        -----

        The expression for :math:`\frac{d \chi}{da}` is:

        .. math::

            \frac{d \chi}{da}(a) = \frac{R_H}{a^2 E(a)}
        """
        return const.rh/(a**2*np.sqrt(self.Esqr(a)))

    def dzoverda(self, a):
        r"""Derivative of the redshift with respect to the scale factor.

        Parameters
        ----------
        a : array_like
            Scale factor

        Returns
        -------
        dz/da :  ndarray, or float if input scalar
            Derivative of the redshift with respect to the scale factor at
            the specified scale factor.

        Notes
        -----
        The expression for :math:`\frac{d z}{da}` is:

        .. math::

            \frac{d z}{da}(a) = \frac{1}{a^2}

        """
        return 1.0 / (a**2)

    def T(self, k, type='eisenhu_osc'):
        """ Computes the matter transfer function.

        Parameters
        ----------
        k: array_like
            Wave number in h Mpc^{-1}

        type: str, optional
            Type of transfer function. Either 'eisenhu' or 'eisenhu_osc'
            (def: 'eisenhu_osc')

        Returns
        -------
        T: array_like
            Value of the transfer function at the requested wave number

        Notes
        -----
        The Eisenstein & Hu transfer functions are computed using the fitting
        formulae of :cite:`1998:EisensteinHu`

        """
        w_m = self.Omega_m * self.h**2
        w_b = self.Omega_b * self.h**2
        fb = self.Omega_b / self.Omega_m
        fc = (self.Omega_m - self.Omega_b) / self.Omega_m

        alpha_gamma = 1.-0.328*np.log(431.*w_m)*w_b/w_m + \
            0.38*np.log(22.3*w_m)*(self.Omega_b/self.Omega_m)**2
        gamma_eff = self.Omega_m*self.h * \
            (alpha_gamma + (1.-alpha_gamma)/(1.+(0.43*k*self.sh_d)**4))

        if(type == 'eisenhu'):

            q = k * np.power(const.tcmb/2.7, 2)/gamma_eff

            # EH98 (29) #
            L = log(2.*exp(1.0) + 1.8*q)
            C = 14.2 + 731.0/(1.0 + 62.5*q)
            res = L/(L + C*q*q)

        elif(type == 'eisenhu_osc'):
            # Cold dark matter transfer function

            # EH98 (11, 12)
            a1 = np.power(46.9*w_m, 0.670) * (1.0 + np.power(32.1*w_m, -0.532))
            a2 = np.power(12.0*w_m, 0.424) * (1.0 + np.power(45.0*w_m, -0.582))
            alpha_c = np.power(a1, -fb) *np.power(a2, -fb**3)
            b1 = 0.944 / (1.0 + np.power(458.0*w_m, -0.708))
            b2 = np.power(0.395*w_m, -0.0266)
            beta_c = 1.0 + b1*(np.power(fc, b2) - 1.0)
            beta_c = 1.0 / beta_c

            # EH98 (19). [k] = h/Mpc
            def T_tilde(k1, alpha, beta):
                # EH98 (10); [q] = 1 BUT [k] = h/Mpc
                q = k1 / (13.41 * self._k_eq)
                L = np.log(np.exp(1.0) + 1.8 * beta * q)
                C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * np.power(q, 1.08))
                T0 = L/(L + C*q*q)
                return T0

            # EH98 (17, 18)
            f = 1.0 / (1.0 + (k * self.sh_d / 5.4)**4)
            Tc = f * T_tilde(k, 1.0, beta_c) + \
                (1.0 - f) * T_tilde(k, alpha_c, beta_c)

            # Baryon transfer function
            # EH98 (19, 14, 21)
            y = (1.0 + self._z_eq) / (1.0 + self._z_d)
            x = np.sqrt(1.0 + y)
            G_EH98 = y * (-6.0 * x +
                          (2.0 + 3.0*y) * np.log((x + 1.0) / (x - 1.0)))
            alpha_b = 2.07 * self._k_eq * self.sh_d * \
                np.power(1.0 + self._R_d, -0.75) * G_EH98

            beta_node = 8.41 * np.power(w_m, 0.435)
            tilde_s = self.sh_d / np.power(1.0 + (beta_node /
                                             (k * self.sh_d))**3, 1.0/3.0)

            beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * np.sqrt((17.2 * w_m)**2 + 1.0)

            # [tilde_s] = Mpc/h
            Tb = (T_tilde(k, 1.0, 1.0) / (1.0 + (k * self.sh_d / 5.2)**2) +
                  alpha_b / (1.0 + (beta_b/(k * self.sh_d))**3) *
                  np.exp(-np.power(k / self._k_silk, 1.4))) * np.sinc(k*tilde_s/np.pi)

            # Total transfer function
            res = fb * Tb + fc * Tc
        else:
            raise NotImplementedError
        return res
    #
    # def G(self, a):
    #     """ Compute Growth factor at a given scale factor, normalised such
    #     that G(a=1) = 1.
    #
    #     Parameters
    #     ----------
    #     a: array_like
    #         Scale factor
    #
    #     Returns
    #     -------
    #     G:  ndarray, or float if input scalar
    #         Growth factor computed at requested scale factor
    #
    #     """
    #
    #     if self._da_interp is None:
    #         def D_derivs(y, x):
    #             q = (2.0 - 0.5 * (self.Omega_m_a(x) +
    #                               (1.0 + 3.0 * self.w(x))
    #                               * self.Omega_de_a(x)))/x
    #             r = 1.5*self.Omega_m_a(x)/x/x
    #             return [y[1], -q * y[1] + r * y[0]]
    #         y0 = [self._amin, 1]
    #
    #         y = odeint(D_derivs, y0, self.atab)
    #         self._da_interp = interp1d(self.atab, y[:, 0], kind='linear')
    #
    #     return self._da_interp(a)/self._da_interp(1.0)

    # def pk_lin(self, k, a=1.0, **kwargs):
    #     r""" Computes the linear matter power spectrum.
    #
    #     Parameters
    #     ----------
    #     k: array_like
    #         Wave number in h Mpc^{-1}
    #
    #     a: array_like, optional
    #         Scale factor (def: 1.0)
    #
    #     type: str, optional
    #         Type of transfer function. Either 'eisenhu' or 'eisenhu_osc'
    #         (def: 'eisenhu_osc')
    #
    #     Returns
    #     -------
    #     pk: array_like
    #         Linear matter power spectrum at the specified scale
    #         and scale factor.
    #
    #     """
    #     k = atleast_1d(k)
    #     a = atleast_1d(a)
    #     g = self.G(a)
    #     t = self.T(k, **kwargs)
    #
    #     pknorm = self.sigma8**2/self.sigmasqr(8.0, **kwargs)
    #
    #     if k.ndim == 1:
    #         pk = outer(self.pk_prim(k) * pow(t, 2), pow(g, 2))
    #     else:
    #         pk = self.pk_prim(k) * pow(t, 2) * pow(g, 2)
    #
    #     # Apply normalisation
    #     pk = pk*pknorm
    #
    #     return pk.squeeze()

    # def _smith_parameters(self, a,  **kwargs):
    #     r""" Computes the non linear scale, effective spectral index
    #     and spectral curvature"""
    #     a = atleast_1d(a)
    #     R_nl = zeros_like(a)
    #     n = zeros_like(a)
    #     C = zeros_like(a)
    #
    #     ksamp = logspace(log10(self._kmin), log10(self._kmax), 1024)
    #     pklog = interp1d(log(ksamp), ksamp**3 *
    #                      self.pk_lin(ksamp, **kwargs) / (2.0*pi**2))
    #     g = self.G(a)
    #
    #     def int_sigma(logk, r, _g):
    #         y = exp(logk)*r
    #         return pklog(logk) * _g**2 * exp(-y**2)
    #
    #     def int_neff(logk, r, _g):
    #         y = exp(logk)*r
    #         return pklog(logk) * _g**2 * y**2 * exp(-y**2)
    #
    #     def int_C(logk, r, _g):
    #         y = exp(logk)*r
    #         return pklog(logk) * _g**2 * (y**2 - y**4) * exp(-y**2)
    #
    #     for i in range(R_nl.size):
    #         sigm = lambda r: romberg(int_sigma, log(self._kmin), log(self._kmax),
    #                                  args=(exp(r), g[i]), rtol=1e-4, vec_func=True) - 1
    #         R_nl[i] = exp(brentq(sigm, -5, 1.5, rtol=1e-4))
    #
    #         n[i] = 2.0 * romberg(int_neff, log(self._kmin), log(self._kmax),
    #                                 args=(R_nl[i], g[i]), rtol=1e-4,  vec_func=True) - 3
    #
    #         C[i] = (3 + n[i])**2 + 4 * romberg(int_C, log(self._kmin), log(self._kmax),
    #                                               args=(R_nl[i], g[i]), rtol=1e-4,  vec_func=True)
    #     k_nl = 1.0/R_nl
    #     return k_nl, n, C
    #
    # def pk(self, k, a=1.0, nl_type='smith2003', **kwargs):
    #     r""" Computes the full non linear matter power spectrum.
    #
    #     Parameters
    #     ----------
    #     k: array_like
    #         Wave number in h Mpc^{-1}
    #
    #     a: array_like, optional
    #         Scale factor (def: 1.0)
    #
    #     nl_type: str, optional
    #         Type of non linear corrections. Only 'smith2003' is implemented
    #
    #     type: str, optional
    #         Type of transfer function. Either 'eisenhu' or 'eisenhu_osc'
    #         (def: 'eisenhu_osc')
    #
    #     Returns
    #     -------
    #     pk: array_like
    #         Non linear matter power spectrum at the specified scale
    #         and scale factor.
    #
    #     Notes
    #     -----
    #     The non linear corrections are implemented following :cite:`2003:smith`
    #
    #     """
    #     k = atleast_1d(k)
    #     a = atleast_1d(a)
    #     pklin = self.pk_lin(k, a, **kwargs)
    #
    #     if (nl_type == 'smith2003'):
    #
    #         # Compute non linear scale, effective spectral index and curvature
    #         k_nl, n, C = self._smith_parameters(a)
    #
    #         om_m = self.Omega_m_a(a)
    #         frac = self.Omega_de_a(a)/(1.0 - om_m)
    #
    #         # eq C9 to C18
    #         a_n = 10**(1.4861 + 1.8369*n + 1.6762*n**2 + 0.7940*n**3 +
    #                    0.1670*n**4 - 0.6206*C)
    #         b_n = 10**(0.9463 + 0.9466*n + 0.3084*n**2 - 0.9400*C)
    #         c_n = 10**(-0.2807 + 0.6669*n + 0.3214*n**2 - 0.0793*C)
    #         gamma_n = 0.8649 + 0.2989*n + 0.1631*C
    #         alpha_n = 1.3884 + 0.3700*n - 0.1452*n**2
    #         beta_n = 0.8291 + 0.9854*n + 0.3401*n**2
    #         mu_n = 10**(-3.5442 + 0.1908*n)
    #         nu_n = 10**(0.9585 + 1.2857*n)
    #
    #         f1a = om_m**(-0.0732)
    #         f2a = om_m**(-0.1423)
    #         f3a = om_m**0.0725
    #         f1b = om_m**(-0.0307)
    #         f2b = om_m**(-0.0585)
    #         f3b = om_m**(0.0743)
    #
    #         f1 = frac*f1b + (1-frac)*f1a
    #         f2 = frac*f2b + (1-frac)*f2a
    #         f3 = frac*f3b + (1-frac)*f3a
    #
    #         f = lambda x: x/4. + x**2/8.
    #
    #         d2l = einsum('i...,i...->i...', k**3, pklin / (2.0*pi**2))
    #         if k.ndim > 1:
    #             y = k/k_nl
    #         else:
    #             y = outer(k, 1.0/k_nl).squeeze()
    #         # Eq C2
    #         d2q = d2l * ((1.0+d2l)**beta_n/(1+alpha_n*d2l)) * exp(-f(y))
    #         d2hprime = a_n*y**(3*f1)/(1.0 + b_n * y**f2 +
    #                                   (c_n*f3*y)**(3.0 - gamma_n))
    #         d2h = d2hprime / (1.0 + mu_n/y + nu_n/y**2)
    #         # Eq. C1
    #         d2nl = d2q + d2h
    #         pk_nl = einsum('i...,i...->i...', 2.0*pi**2/k**3, d2nl)
    #     else:
    #         print("unknown non linear prescription")
    #         pk_nl = pklin
    #
    #     return pk_nl.squeeze()
    #
    # def pl(self, l, a, **kwargs):
    #     r"""
    #     Computes the non linear matter power spectrum at a given angular scale
    #     using the Limber approximation
    #
    #     """
    #     k = outer(l + 0.5, 1.0/self.a2chi(a))
    #
    #     return self.pk(k, a, **kwargs)
    #
    # def pl_lin(self, l, a, **kwargs):
    #     """
    #     Computes the linear matter power spectrum at the specified scale
    #     and scale factor
    #     """
    #
    #     k = outer(l + 0.5, 1.0/self.a2chi(a))
    #
    #     g = self.G(a)
    #     t = self.T(k, **kwargs)
    #
    #     pknorm = self.sigma8**2/self.sigmasqr(8.0, **kwargs)
    #
    #     pk = multiply(self.pk_prim(k) * pow(t, 2), pow(g, 2))
    #
    #     # Apply normalisation
    #     pk = pk*pknorm
    #
    #     return pk

    def pk_prim(self, k):
        """ Primordial power spectrum
            Pk = k^n
        """
        return k**self.n

    # def sigmasqr(self, R, **kwargs):
    #     """ Computes the energy of the fluctuations within a sphere of R h^{-1} Mpc
    #
    #     .. math::
    #
    #        \\sigma^2(R)= \\frac{1}{2 \\pi^2} \\int_0^\\infty \\frac{dk}{k} k^3 P(k,z) W^2(kR)
    #
    #     where
    #
    #     .. math::
    #
    #        W(kR) = \\frac{3j_1(kR)}{kR}
    #     """
    #     def int_sigma(logk):
    #         k = exp(logk)
    #         x = k * R
    #         w = 3.0*(sin(x) - x*cos(x))/(x*x*x)
    #         pk = self.T(k, **kwargs)**2 * self.pk_prim(k)
    #
    #         return k * pow(k*w, 2.0) * pk
    #
    #     return 1.0/(2.0*pi**2.0) * romberg(int_sigma, log(self._kmin),
    #                                        log(self._kmax))
    #
    # def g(self, a, a_s):
    #     """ Lensing efficiency kernel computed a distance chi for sources
    #     placed at distance chi_s
    #     """
    #     a_s = atleast_1d(a_s)
    #     a   = atleast_1d(a)
    #     factor = 3.0 * const.H0**2 * self.Omega_m / (2.0 * const.c**2)/a
    #     res = (self.a2chi(a_s) - self.a2chi(a)) / self.a2chi(a_s)
    #     res[a_s > a] = 0
    #
    #     return factor * self.f_k(a) * res * self.dchioverda(a)

    # Derived constants
    @property
    def Omega_dm(self):
        return self._Omega_dm
    @property
    def Omega(self):
        return self._Omega
    @property
    def Omega_k(self):
        return self._Omega_k
    @property
    def gamma(self):
        return self._gamma
    @property
    def k(self):
        return self._k
    @property
    def sqrtk(self):
        return self._sqrtk

    @property
    def sh_d(self):
        r"""
        Sound horizon at drag epoch in Mpc/h

        Computed from Equation (6) in :cite:`1998:EisensteinHu` :

        .. math ::

            r_s(z_d) = \frac{2}{3 k_{eq}} \sqrt{ \frac{6}{R_{eq}} } \ln \frac{ \sqrt{1 + R_d} + \sqrt{R_d + R_{eq}}}{1 + \sqrt{R_{eq}}}

        where :math:`R_d` and :math:`R_{eq}` are respectively the ratio of baryon to photon momentum density at drag epoch and equality epoch (see Equation (5) in :cite:`1998:EisensteinHu`)
        and :math:`k_{eq}` is the scale of the scale of the particle horizon at equality epoch.
        """
        return self._sh_d

    @property
    def sh_r(self):
        r"""
        Sound horizon at recombination in Mpc/h

        Computed from Equation (19) in :cite:`1999:EfstathiouBond` :

        .. math ::

            r_s(z_r) = \frac{4000 \sqrt{a_{equ}}}{\sqrt{\omega_b (1 + \eta_\nu)}} \ln \frac{ \sqrt{1 + R_r} + \sqrt{R_r + R_{eq}}}{1 + \sqrt{R_{eq}}}

        where :math:`R_r` and :math:`R_{eq}` are respectively the ratio of baryon to photon momentum density at recombination epoch and equality epoch (see Equation (18) in :cite:`1999:EfstathiouBond`)
        and :math:`\eta_{\nu}` denotes the relative densities of massless neutrinos and photons.
        """
        return self._sh_r

    # Settable cosmological parameters
    @property
    def h(self):
        return self._h

    @h.setter
    def h(self,val):
        self.update(h=val)


    @property
    def Omega_b(self):
        """
            baryon density
        """
        return self._Omega_b
    @Omega_b.setter
    def Omega_b(self,val):
        self.update(Omega_b=val)

    @property
    def Omega_m(self):
        return self._Omega_m
    @Omega_m.setter
    def Omega_m(self,val):
        self.update(Omega_m = val)

    @property
    def Omega_de(self):
        return self._Omega_de
    @Omega_de.setter
    def Omega_de(self,val):
        self.update(Omega_de = val)

    @property
    def w0(self):
        return self._w0
    @w0.setter
    def w0(self,val):
        self.update(w0 = val)

    @property
    def wa(self):
        return self._wa
    @wa.setter
    def wa(self,val):
        self.update(wa=val)

    @property
    def n(self):
        return self._n
    @n.setter
    def n(self,val):
        self.update(n=val)

    @property
    def tau(self):
        return self._tau
    @tau.setter
    def tau(self, val):
        self.update(tau=val)

    @property
    def sigma8(self):
        return self._sigma8
    @sigma8.setter
    def sigma8(self, val):
        self.update(sigma8=val)
