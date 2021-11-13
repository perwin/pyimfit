Description Classes
===================

The are classes which hold information about individual instances of Imfit image functions
(FunctionDescription), the function sets which group one or more functions with a common central
coordinate (FunctionSet), and finally Models which combine all function sets with optional
image-description information (ModelDescription, SimpleModelDescription).

Each FunctionDescription, in turn, holds a set of ParameterDescription objects,
which keep track of parameter names, initial or current values, and limits.

To properly instantiate an instance of the Imfit class, you must supply it with a previously
generated instance of the ModelDescription (or SimpleModelDescription) class.


.. automodule:: pyimfit.descriptions
      :imported-members:
      :members:
