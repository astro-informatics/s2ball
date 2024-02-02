:html_theme.sidebar_secondary.remove:

**************************
Wavelet Functions
**************************

.. list-table:: Generating Functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.wavelets.kernels.tiling_integrand`
     - Tiling integrand for scale-discretised wavelets.
   * - :func:`~s2ball.wavelets.kernels.part_scaling_fn`
     - Computes Infinitely differentiable Cauchy-Schwartz function.
   * - :func:`~s2ball.wavelets.kernels.k_lam`
     - Computes wavelet generating function.
   * - :func:`~s2ball.wavelets.kernels.tiling_direction`
     - Generates harmonic coefficients for directional tiling functions.
   * - :func:`~s2ball.wavelets.kernels.binomial_coefficient`
     - Computes the binomial coefficient :math:`\binom{n}{k}`.

.. list-table:: Wavelet Filters
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.wavelets.tiling.construct_wav_lmp`
     - Generate multiresolution wavelet filters.
   * - :func:`~s2ball.wavelets.tiling.construct_f_wav_lmnp`
     - Generate multiresolution wavelet Wigner-Laguerre coefficients for Numpy.
   * - :func:`~s2ball.wavelets.tiling.construct_f_wav_lmnp_jax`
     - Generate multiresolution wavelet Wigner-Laguerre coefficients for JAX.
   * - :func:`~s2ball.wavelets.tiling.construct_wav_lmp`
     - Compute multiresolution wavelet and scaling filters.
   * - :func:`~s2ball.wavelets.tiling.tiling_axisym`
     - Generates tuple of axisymmetric tiling functions.

.. list-table:: Helper Functions
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.wavelets.helper_functions.j_max`
     - Computes the highest wavelet scale :math:`J_{\text{max}}`.
   * - :func:`~s2ball.wavelets.helper_functions.radial_bandlimit`
     - Computes the radial band-limit for scale :math:`j_p`.
   * - :func:`~s2ball.wavelets.helper_functions.angular_bandlimit`
     - Computes the angular band-limit for scale :math:`j_{\ell}`.
   * - :func:`~s2ball.wavelets.helper_functions.wavelet_scale_limits`
     - Computes the angular and radial band-limits for scale :math:`j_{\ell}/j_p`.
   * - :func:`~s2ball.wavelets.helper_functions.wavelet_scale_limits_N`
     - Computes the angular and radial band-limits and multiresolution directionality :math:`N_j` for scale :math:`j_{\ell}/j_p`.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Wavelet Functions

   kernels
   tiling
   helper_functions