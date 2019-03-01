The Imfit class
=============================

The core of PyImfit is the Imfit class, which acts as a wrapper around the
underlying C++ ModelObject instance. It holds data, PSF images, image parameters
(A/D gain, etc.), and a ModelDescription instance which describes the model to
be fit to the data.

It also has methods for running a fit and for inspecting the results.

.. automodule:: pyimfit.fitting
      :members:
