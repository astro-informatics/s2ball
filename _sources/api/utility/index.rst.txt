:html_theme.sidebar_secondary.remove:

**************************
Utility Functions
**************************

.. list-table:: Signal Generators
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.utils.generate_flm`
     - Function which generates a 2D set of random harmonic coefficients.
   * - :func:`~s2ball.utils.generate_flmn`
     - Function which generates a 3D set of random Wigner coefficients.
   * - :func:`~s2ball.utils.generate_flmp`
     - Function which generates a 3D set of random Spherical-Laguerre coefficients.
   * - :func:`~s2ball.utils.generate_flmnp`
     - Function which generates a 4D set of random Wigner-Laguerre coefficients.

.. list-table:: Indexing Functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.utils.ncoeff`
     - Computes the number of spherical harmonic coefficients for given band-limit L.
   * - :func:`~s2ball.utils.elm2ind`
     - Converts from spherical harmonic 2D indexing of :math:`(\ell,m)` to 1D index.
   * - :func:`~s2ball.utils.ind2elm`
     - Converts from 1D spherical harmonic index to 2D index of :math:`(\ell,m)`.
   * - :func:`~s2ball.utils.elmn2ind`
     - Converts from Wigner space 3D indexing of :math:`(\ell,m, n)` to 1D index.
   * - :func:`~s2ball.utils.flm_2d_to_1d`
     - Converts from 2D indexed harmonic coefficients to 1D indexed coefficients.
   * - :func:`~s2ball.utils.flm_1d_to_2d`
     - Converts from 1D indexed harmnonic coefficients to 2D indexed coefficients.
   * - :func:`~s2ball.utils.flmn_3d_to_1d`
     - Converts from 3D indexed Wigner coefficients to 1D indexed coefficients.
   * - :func:`~s2ball.utils.flmn_1d_to_3d`
     - Converts from 1D indexed Wigner coefficients to 3D indexed coefficients.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Utility Functions

   utils
   