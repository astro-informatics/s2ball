:html_theme.sidebar_secondary.remove:

**************************
Notebooks
**************************
A series of tutorial notebooks which go through the absolute base level application of ``S2BALL`` apis. Post alpha release we will add examples for more involved applications, in the time being feel free to contact contributors for advice! At a high-level the ``S2BALL`` package is structured such that the 4 primary transforms: the Wigner, spherical harmonic, Fourier-Laguerre, and Wigner-Laguerre transforms, can easily be accessed.

Usage |:rocket:|
-----------------
To apply the transforms provided by ``S2BALL`` one need only import the package and apply the respective transform, which is as simple as doing the following: 

.. code-block:: Python

    from s2ball.transform import *
    import numpy as np 

    # Load some data
    f = np.load("path_to_your_data.npy")

    # Select your method: JAX is recommended even on CPU for JIT compilation.
    alg = ["numpy", "jax"]

+-------------------------------------------------------+------------------------------------------------------------+
|and for data on the sphere with shape :math:`[L, 2L-1]`|or data on SO(3) with shape :math:`[2N-1, L, 2L-1]`         |
|                                                       |                                                            |
|.. code-block:: Python                                 |.. code-block:: Python                                      |
|                                                       |                                                            |
|   L = L                                               |   L = L; N = N                                             |
|                                                       |                                                            |
|   # Compute harmonic coefficients                     |   # Compute Wigner coefficients                            |
|   flm = harmonic.forward(f, L, alg)                   |   flmn = wigner.forward(f, L, N, alg)                      |
|                                                       |                                                            |
|   # Sythensise signal f                               |   # Sythensise signal f                                    |
|   f = harmonic.inverse(flm, L, alg)                   |   f = wigner.inverse(flmn, L, N, alg)                      |
+-------------------------------------------------------+------------------------------------------------------------+

+---------------------------------------------------+---------------------------------------------------------+
|or data on the ball with shape :math:`[P, L, 2L-1]`|or with shape :math:`[P, 2N-1, L, 2L-1]`                 |
|                                                   |                                                         |
|.. code-block:: Python                             |.. code-block:: Python                                   |
|                                                   |                                                         |
|   L = L; P = P                                    |   L = L; N = N; P = P                                   |
|                                                   |                                                         |
|   # Compute spherical-Laguerre coefficients       |   # Compute Wigner coefficients                         |
|   flmp = laguerre.forward(f, L, P, alg)           |   flmnp = wigner_laguerre.forward(f, L, N, P, alg)      |
|                                                   |                                                         |
|   # Sythensise signal f                           |   # Sythensise signal f                                 |
|   f = laguerre.inverse(flmp, L, P, alg)           |   f = wigner_laguerre.inverse(flmnp, L, N, P, alg)      |
+---------------------------------------------------+---------------------------------------------------------+

However, for repeated application of these transforms it is optimal to instead precompile 
various kernels which can be placed on device to minimise i/o during *e.g.* training. This 
operational mode can be seen throughout our examples, found `here 
<https://github.com/astro-informatics/s2ball/tree/main/notebooks>`_.

Benchmarking |:hourglass_flowing_sand:|
-------------------------------------
The various generalized Fourier and wavelet transforms supported by ``S2BALL`` were 
benchmarked against their ``C`` counterparts over a variety of parameter configurations. 
Each benchmark has been averaged over many runs, though here we provide only the mean. 
All CPU based operations were executed on a single core from a AMD EPYC 7702 64-core 
processor, whereas all JAX operations were executed on a single NVIDIA A100 graphics 
processing unit. The Jupyter notebooks for each benchmark can be found `here 
<https://github.com/astro-informatics/s2ball/tree/main/notebooks>`_.

Note that benchmarking is restricted to scalar (spin 0 ) fields, though spin is supported 
throughout ``S2BALL``. Further note that for Wigner tests we set N=5, and in our 
Laguerre and wavelet benchmarking we set N=1, as FLAG/FLAGLET otherwise take 
excessive compute. Finally, ``S2BALL``` transforms trivially support batching and 
so can, in many cases, gain several more orders of magnitude acceleration.
    
|harmonic| |wigner| 

|laguerre| |wavelet|

.. |harmonic| image:: ../assets/figures/harmonic.png
    :width: 48%

.. |wigner| image:: ../assets/figures/wigner.png
    :width: 48%

.. |laguerre| image:: ../assets/figures/laguerre.png
    :width: 48%

.. |wavelet| image:: ../assets/figures/wavelet.png
    :width: 48%

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks

   spherical_harmonic/spherical_harmonic_transform.nblink
   wigner/wigner_transform.nblink
   laguerre/laguerre_transform.nblink
   wigner_laguerre/wigner_laguerre_transform.nblink
   wavelet/wavelet_transform.nblink
