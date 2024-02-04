:html_theme.sidebar_secondary.remove:

**************************
Transforms
**************************
Note that all transforms straightforwardly provide support for adjoint transforms through a single function variable.

.. list-table:: Spherical harmonic transforms
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.transform.harmonic.forward`
     - Wrapper function for forward spherical harmonic transform.
   * - :func:`~s2ball.transform.harmonic.forward_transform`
     - Compute the forward spherical harmonic transform with Numpy.
   * - :func:`~s2ball.transform.harmonic.forward_transform_jax`
     - Compute the forward spherical harmonic transform with JAX.
   * - :func:`~s2ball.transform.harmonic.inverse`
     - Wrapper function for inverse spherical harmonic transform.
   * - :func:`~s2ball.transform.harmonic.inverse_transform`
     - Compute the inverse spherical harmonic transform with Numpy.
   * - :func:`~s2ball.transform.harmonic.inverse_transform_jax`
     - Compute the inverse spherical harmonic transform with JAX.

.. list-table:: Wigner transforms
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.transform.wigner.forward`
     - Wrapper function for forward Wigner transform.
   * - :func:`~s2ball.transform.wigner.forward_transform`
     - Compute the forward Wigner transform with Numpy.
   * - :func:`~s2ball.transform.wigner.forward_transform_jax`
     - Compute the forward Wigner transform with JAX.
   * - :func:`~s2ball.transform.wigner.inverse`
     - Wrapper function for inverse Wigner transform.
   * - :func:`~s2ball.transform.wigner.inverse_transform`
     - Compute the inverse Wigner transform with Numpy.
   * - :func:`~s2ball.transform.wigner.inverse_transform_jax`
     - Compute the inverse Wigner transform with JAX.

.. list-table:: Spherical-Laguerre transforms
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.transform.laguerre.forward`
     - Wrapper function for forward Spherical-Laguerre transform.
   * - :func:`~s2ball.transform.laguerre.forward_transform`
     - Compute the forward Spherical-Laguerre transform with Numpy.
   * - :func:`~s2ball.transform.laguerre.forward_transform_jax`
     - Compute the forward Spherical-Laguerre transform with JAX.
   * - :func:`~s2ball.transform.laguerre.inverse`
     - Wrapper function for inverse Spherical-Laguerre transform.
   * - :func:`~s2ball.transform.laguerre.inverse_transform`
     - Compute the inverse Spherical-Laguerre transform with Numpy.
   * - :func:`~s2ball.transform.laguerre.inverse_transform_jax`
     - Compute the inverse Spherical-Laguerre transform with JAX.

.. list-table:: Wigner-Laguerre transforms
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.transform.wigner_laguerre.forward`
     - Wrapper function for forward Wigner-Laguerre transform.
   * - :func:`~s2ball.transform.wigner_laguerre.forward_transform`
     - Compute the forward Wigner-Laguerre transform with Numpy.
   * - :func:`~s2ball.transform.wigner_laguerre.forward_transform_jax`
     - Compute the forward Wigner-Laguerre transform with JAX.
   * - :func:`~s2ball.transform.wigner_laguerre.inverse`
     - Wrapper function for inverse Wigner-Laguerre transform.
   * - :func:`~s2ball.transform.wigner_laguerre.inverse_transform`
     - Compute the inverse Wigner-Laguerre transform with Numpy.
   * - :func:`~s2ball.transform.wigner_laguerre.inverse_transform_jax`
     - Compute the inverse Wigner-Laguerre transform with JAX.

.. list-table:: Wavelet transforms
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2ball.transform.ball_wavelet.forward`
     - Wrapper function for forward ball wavelet transform.
   * - :func:`~s2ball.transform.ball_wavelet.forward_transform`
     - Compute the forward ball wavelet transform with Numpy.
   * - :func:`~s2ball.transform.ball_wavelet.forward_transform_jax`
     - Compute the forward ball wavelet transform with JAX.
   * - :func:`~s2ball.transform.ball_wavelet.inverse`
     - Wrapper function for inverse ball wavelet transform.
   * - :func:`~s2ball.transform.ball_wavelet.inverse_transform`
     - Compute the inverse ball wavelet transform with Numpy.
   * - :func:`~s2ball.transform.ball_wavelet.inverse_transform_jax`
     - Compute the inverse ball wavelet transform with JAX.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Transforms

   harmonic
   wigner
   laguerre
   wigner_laguerre
   wavelet
   