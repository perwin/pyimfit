The Imfit class
=============================

The core of PyImfit is the Imfit class, which acts as a wrapper around the
underlying C++ ModelObject instance. It holds a ModelDescription instance which
describes the model to be fit to the data (or just used to generate a model
image, if no actual fitting is to be done). In the (usual) case of fitting the
model to an image, it also holds the data image, optional PSF images, and
parameters that describe the image (A/D gain, etc.).

It also has methods for running a fit and for inspecting the results.

.. automodule:: pyimfit.fitting
      :members:
