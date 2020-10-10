/** @file
 * \brief Code for calling specific solvers (minimization algorithms)
 *
 * 
 *
 */

#ifndef _DISPATCH_SOLVER_H_
#define _DISPATCH_SOLVER_H_

#include <string>
#include "model_object.h"
#include "param_struct.h"   // for mp_par structure
#include "solver_results.h"

// NOTE: The following functions is used in PyImfit

// Possible return values from DispatchToSolver
// All:
// <= 0 = error of some kind
// 1 through 4: successful fit
// 5+: maximum iterations/neval or similar
//
// NLOpt: value of type nlopt_result 
// 1, 3, or 4 = successful fit
// 2, 5, or 6 = hit chi^2 limit, ran out of iterations
//
// typedef enum {
//     NLOPT_FAILURE = -1,         /* generic failure code */
//     NLOPT_INVALID_ARGS = -2,
//     NLOPT_OUT_OF_MEMORY = -3,
//     NLOPT_ROUNDOFF_LIMITED = -4,
//     NLOPT_FORCED_STOP = -5,
//     NLOPT_SUCCESS = 1,          /* generic success code */
//     NLOPT_STOPVAL_REACHED = 2,  -- PE: not relevant
//     NLOPT_FTOL_REACHED = 3,
//     NLOPT_XTOL_REACHED = 4,
//     NLOPT_MAXEVAL_REACHED = 5,
//     NLOPT_MAXTIME_REACHED = 6
// } nlopt_result;
//
// LM:
// 1--4,6--8 = successful fit
// 5 = ran out of iterations
//
// /* Error codes */
// #define MP_ERR_INPUT (0)         /* General input parameter error */
// #define MP_ERR_NAN (-16)         /* User function produced non-finite values */
// #define MP_ERR_FUNC (-17)        /* No user function was supplied */
// #define MP_ERR_NPOINTS (-18)     /* No user data points were supplied */
// #define MP_ERR_NFREE (-19)       /* No free parameters */
// #define MP_ERR_MEMORY (-20)      /* Memory allocation error */
// #define MP_ERR_INITBOUNDS (-21)  /* Initial values inconsistent w constraints*/
// #define MP_ERR_BOUNDS (-22)      /* Initial constraints inconsistent */
// #define MP_ERR_PARAM (-23)       /* General input parameter error */
// #define MP_ERR_DOF (-24)         /* Not enough degrees of freedom */
// 
// /* Potential success status codes */
// #define MP_OK_CHI (1)            /* Convergence in chi-square value */
// #define MP_OK_PAR (2)            /* Convergence in parameter value */
// #define MP_OK_BOTH (3)           /* Both MP_OK_PAR and MP_OK_CHI hold */
// #define MP_OK_DIR (4)            /* Convergence in orthogonality */
// #define MP_MAXITER (5)           /* Maximum number of iterations reached */
// #define MP_FTOL (6)              /* ftol is too small; no further improvement*/
// #define MP_XTOL (7)              /* xtol is too small; no further improvement*/
// #define MP_GTOL (8)              /* gtol is too small; no further improvement*/
//
// DE:
//    value < 0   --> FAILURE
//    value = 0   --> FAILURE: input parameter error
//    value = 1   --> generic success
//    value = 5   --> max iterations reached

/// Function which handles selecting and calling appropriate solver
int DispatchToSolver( int solverID, int nParametersTot, int nFreeParameters, int nPixelsTot,
					double *parameters, vector<mp_par> parameterInfo, ModelObject *modelObj, 
					double fracTolerance, bool paramLimitsExist, int verboseLevel, 
					SolverResults *solverResults, string& solverName, 
					unsigned long rngSeed=0, bool useLHS=false );


#endif /* _DISPATCH_SOLVER_H_ */
