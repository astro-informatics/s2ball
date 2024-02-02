:html_theme.sidebar_secondary.remove:

**************************
Matrix Bootstrapping
**************************

.. list-table:: Associated Legendre Matrices
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.construct.legendre_constructor.construct_legendre_matrix`
     - Constructs associated Legendre matrix which will be called during transform.
   * - :func:`~s2ball.construct.legendre_constructor.construct_legendre_matrix_inverse`
     - Constructs associated Legendre matrix which will be called during inverse transform.
   * - :func:`~s2ball.construct.legendre_constructor.compute_legendre_warning`
     - Basic compute warning for large Legendre precomputes.
   * - :func:`~s2ball.construct.legendre_constructor.load_legendre_matrix`
     - Load/construct associated Legendre inverse matrix for precompute method.

.. list-table:: Wigner Matrices
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.construct.wigner_constructor.construct_wigner_matrix`
     - Constructs Wigner matrix which will be called during transform.
   * - :func:`~s2ball.construct.wigner_constructor.construct_wigner_matrix_inverse`
     - Constructs Wigner matrix which will be called during inverse transform.
   * - :func:`~s2ball.construct.wigner_constructor.compute_wigner_warning`
     - Basic compute warning for large Wigner precomputes.
   * - :func:`~s2ball.construct.wigner_constructor.load_wigner_matrix`
     - Load/construct Wigner inverse matrix for precompute method.
    
.. list-table:: Wavelet Matrices
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.construct.wavelet_constructor.wavelet_legendre_kernels`
     - Constructs an array which holds associated Legendre matrices for the forward and inverse spherical harmonic transforms respectively.
   * - :func:`~s2ball.construct.wavelet_constructor.scaling_laguerre_kernels`
     - Constructs and array which holds the Laguerre polynomials for forward and inverse Fourier-Laguerre transforms respectively.
   * - :func:`~s2ball.construct.wavelet_constructor.wavelet_wigner_kernels`
     - Constructs a collection of Wigner kernels for multiresolution directional wavelet transforms.
   * - :func:`~s2ball.construct.wavelet_constructor.wavelet_laguerre_kernels`
     - Constructs a collection of Laguerre polynomial kernel for multiresolution directional wavelet transforms.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Matrix Bootstrapping

   legendre_constructor
   wigner_constructor
   wavelet_constructor
   