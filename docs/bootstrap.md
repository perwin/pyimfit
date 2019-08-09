# Bootstrap Resampling for Parameter Uncertainties

TBD.

## Bootstrap Resampling

When you call the `doFit` method on an Imfit object with the default Levenberg-Marquardt
solver, the solution automatically involves estimation of uncertainties on the best-fit
parameters, in the form of 1-sigma Gaussian errors. XXX

A somewhat better (albeit more time-consuming) approach is to estimate the parameter
uncertainties -- including possible correlations between the values of different
parameters -- via bootstrap resampling.

## Using MCMC

Estimates of parameter uncertainties and correlations can also be obtained via
Markov-Chain Monte Carlo (MCMC) approaches. Although the MCMC option of **Imfit** (`imfit-mcmc`)
is not part of PyImfit, you *can* use instances of the Imfit class
with Python-based MCMC codes, such as [emcee](https://github.com/dfm/emcee); 
see [here](./pyimfit_emcee.html) for an example.
