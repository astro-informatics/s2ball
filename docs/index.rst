Differentiable and accelerated wavelets on the ball
============================================================

``S2BALL`` is a ``JAX`` package for computing the scale-discretised wavelet transform on the ball and rotational ball `(Price et al 2024) <https://arxiv.org/abs/2311.14670>`_. It leverages autodiff to provide differentiable transforms, which are also deployable on modern hardware accelerators (e.g. GPUs and TPUs). See `Price & McEwen 2023 <https://arxiv.org/abs/2311.14670>`_ for similar work which provides generalised Fourier transforms on the sphere and rotation group.

The transforms ``S2BALL`` provides are optimally fast but come with a substantial memory 
overhead and cannot be used above a harmonic bandlimit of L ~ 256, at least with current GPU memory 
limitations. That being said, many applications are more than comfortable at these resolutions, for 
which these ``JAX`` transforms are ideally suited, *e.g.* geophysical modelling, diffusion 
tensor imaging, multiscale molecular modelling. For those with machine learning in mind, 
it should be explicitly noted that these transforms are indeed equivariant to their respective groups.

Wavelet Filters |:ringed_planet:|
----------------------------------------------

The filters ``S2BALL`` provides were originally derived by `Leistedt & McEwen 2012 
<https://arxiv.org/pdf/1205.0792.pdf>`_ and are constructed by tesselating both harmonic space 
and the radial half-line with infinitely differentiable Cauchy-Schwartz functions. This tesselation 
gives rise to the follow frequency space localisation 

|filter_support|

.. |filter_support| image:: ./assets/figures/ball_filter_support.png
    :width: 90%

The above is the frequency space localisation of the wavelet filters, however one can also view wavelet filters in pixel space. Visualising these filters is somewhat tricky as the ball is a 3-dimensional surface embedded in 4-dimensional space. We can, however, straightforwardly view a spherical slice of the ball for each radial node

|filter_support_pixel|

.. |filter_support_pixel| image:: ./assets/figures/ball_filter_support_pixelspace.png
    :width: 90%


Attribution |:books:|
---------------------

Should this code be used in any way, we kindly request that the following article is
referenced. A BibTeX entry for this reference may look like:

.. code-block:: 

    @article{price:s2ball, 
        author      = "Matthew A. Price and Alicja Polanska and Jessica Whitney and Jason D. McEwen",
        title       = "Differentiable and accelerated directional wavelet transform on the sphere and ball",
        journal     = "The Open Journal of Astrophysics, submitted",
        year        = "2024",
        eprint      = "arXiv:0000.0000"        
    }

This work is provided as part of a collection of `JAX` harmonic analysis packages which include 

.. code-block:: 

    @article{price:s2fft, 
        author      = "Matthew A. Price and Jason D. McEwen",
        title       = "Differentiable and accelerated spherical harmonic and Wigner transforms",
        journal     = "Journal of Computational Physics, submitted",
        year        = "2023",
        eprint      = "arXiv:2311.14670"        
    }
    
You might also like to consider citing our related papers on which this code builds:

.. code-block:: 

    @article{leistedt:flaglets,
        author      = "Boris Leistedt and Jason D. McEwen",
        title       = "Exact wavelets on the ball",
        journal     = "IEEE Trans. Sig. Proc.",
        year        = "2012",
        volume      = "60",
        number      = "12",
        pages       = "6257--6269",        
        eprint      = "arXiv:1205.0792",
        doi         = "110.1109/TSP.2012.2215030"
    }

.. code-block:: 

    @article{mcewen:fssht,
        author      = "Jason D. McEwen and Yves Wiaux",
        title       = "A novel sampling theorem on the sphere",
        journal     = "IEEE Trans. Sig. Proc.",
        year        = "2011",
        volume      = "59",
        number      = "12",
        pages       = "5876--5887",        
        eprint      = "arXiv:1110.6298",
        doi         = "10.1109/TSP.2011.2166394"
    }

.. code-block:: 

    @article{mcewen:so3,
        author      = "Jason D. McEwen and Martin B{\"u}ttner and Boris ~Leistedt and Hiranya V. Peiris and Yves Wiaux",
        title       = "A novel sampling theorem on the rotation group",
        journal     = "IEEE Sig. Proc. Let.",
        year        = "2015",
        volume      = "22",
        number      = "12",
        pages       = "2425--2429",
        eprint      = "arXiv:1508.03101",
        doi         = "10.1109/LSP.2015.2490676"    
    }

License |:memo:|
----------------

``S2BALL`` is released under the MIT license (see `LICENSE.txt <https://github.com/astro-informatics/s2ball/blob/main/LICENCE.txt>`_).

.. code-block:: bash

    We provide this code under an MIT open-source licence with the hope that it will be of use to a wider community.
    Copyright 2024 Matthew Price, Jason McEwen and contributors.
    S2BALL is free software made available under the MIT License. For details see the LICENSE file.

.. bibliography:: 
    :notcited:
    :list: bullet

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/install

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Interactive Tutorials
   
   tutorials/index

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API

   api/index