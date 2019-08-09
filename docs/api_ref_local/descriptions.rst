Description Classes
===================

The are classes which hold information about individual instances of Imfit image functions
(FunctionDescription), the function sets which group one or more functions with a common central
coordinate (FunctionSet), and finally Models which combine all function sets with optional
image-description information (ModelDescription, SimpleModelDescription).

Each FunctionDescription, in turn, holds a set of ParameterDescription objects,
which keep track of parameter names, values, and limits.

.. automodule:: pyimfit.descriptions
      :members:
