/** @file
 * \brief Code implementing Levenberg-Marquardt minimization, based on Craig Markwardt's mpfit
 *
 * MINPACK-1 Least Squares Fitting Library
 *
 * Original public domain version by B. Garbow, K. Hillstrom, J. More'
 *   (Argonne National Laboratory, MINPACK project, March 1980)
 * 
 * Tranlation to C Language by S. Moshier (moshier.net)
 * 
 * Enhancements and packaging by C. Markwardt
 *   (comparable to IDL fitting routine MPFIT
 *    see http://cow.physics.wisc.edu/~craigm/idl/idl.html)
 */

/* Minor modifications by Peter Erwin:
 *     1 Oct 2010: Added verbose flag to mp_config_struct
 *     7 Apr 2010: Moved mp_par_struct definition to param_struct.h
 *     5 Apr 2010: Added InterpretMpfitResult() function.
 *     5 Dec 2009: Added #ifndef wrapper
 *    14 Nov 2009: changed "private" to "privateData" ("private" is C++ keyword,
 * making it difficult to use this with a C++ program); included model_object.h
 * header so we can refer to ModelObject class.
 */

/* Header file defining constants, data structures and functions of
   mpfit library 
   $Id: mpfit.h,v 1.9 2009/02/18 23:08:49 craigm Exp $
*/

#ifndef _MPFIT_H_
#define _MPFIT_H_

#include <string>
#include "param_struct.h"
#include "model_object.h"


/* MPFIT version string */
#define MPFIT_VERSION "1.1"


/* Just a placeholder - do not use!! */
typedef void (*mp_iterproc)(void);

/* Definition of MPFIT configuration structure */
struct mp_config_struct {
  double ftol;    /* Relative chi-square convergence criterium */
  double xtol;    /* Relative parameter convergence criterium */
  double gtol;    /* Orthogonality convergence criterium */
  double epsfcn;  /* Finite derivative step size */
  double stepfactor; /* Initial step bound */
  double covtol;  /* Range tolerance for covariance calculation */
  int maxiter;    /* Maximum number of iterations.  If maxiter == 0,
                     then basic error checking is done, and parameter
                     errors/covariances are estimated based on input
                     parameter values, but no fitting iterations are done. */
  int maxfev;     /* Maximum number of function evaluations */
  int nprint;
  int douserscale;/* Scale variables by user values?
		     1 = yes, user scale values in diag;
		     0 = no, variables scaled internally */
  int nofinitecheck; /* Disable check for infinite quantities from user?
			0 = do not perform check
			1 = perform check 
		     */
  mp_iterproc iterproc; /* Placeholder pointer - must set to 0 */
  int  verbose;

};


// NOTE: (parts of) the following struct are used in PyImfit

/* Definition of results structure, for when fit completes */
struct mp_result_struct {
  double bestnorm;     /* Final chi^2 */
  double orignorm;     /* Starting value of chi^2 */
  int niter;           /* Number of iterations */
  int nfev;            /* Number of function evaluations */
  int status;          /* Fitting status code */
  
  int npar;            /* Total number of parameters */
  int nfree;           /* Number of free parameters */
  int npegged;         /* Number of pegged parameters */
  int nfunc;           /* Number of residuals (= num. of data points) */

  double *resid;       /* Final residuals
			  nfunc-vector, or 0 if not desired */
  double *xerror;      /* Final parameter uncertainties (1-sigma)
			  npar-vector, or 0 if not desired */
  double *covar;       /* Final parameter covariance matrix
			  npar x npar array, or 0 if not desired */
};  

/* Convenience typedefs */  
typedef struct mp_config_struct mp_config;
typedef struct mp_result_struct mp_result;

/* Enforce type of fitting function */
typedef int (*mp_func)(int m, /* Number of functions (elts of fvec) */
		       int n, /* Number of variables (elts of x) */
		       double *x,      /* I - Parameters */
		       double *fvec,   /* O - function values */
		       double **dvec,  /* O - function derivatives (optional)*/
		       ModelObject *theModel); /* I/O - function private data*/

/* Error codes */
#define MP_ERR_INPUT (0)         /* General input parameter error */
#define MP_ERR_NAN (-16)         /* User function produced non-finite values */
#define MP_ERR_FUNC (-17)        /* No user function was supplied */
#define MP_ERR_NPOINTS (-18)     /* No user data points were supplied */
#define MP_ERR_NFREE (-19)       /* No free parameters */
#define MP_ERR_MEMORY (-20)      /* Memory allocation error */
#define MP_ERR_INITBOUNDS (-21)  /* Initial values inconsistent w constraints*/
#define MP_ERR_BOUNDS (-22)      /* Initial constraints inconsistent */
#define MP_ERR_PARAM (-23)       /* General input parameter error */
#define MP_ERR_DOF (-24)         /* Not enough degrees of freedom */

/* Potential success status codes */
#define MP_OK_CHI (1)            /* Convergence in chi-square value */
#define MP_OK_PAR (2)            /* Convergence in parameter value */
#define MP_OK_BOTH (3)           /* Both MP_OK_PAR and MP_OK_CHI hold */
#define MP_OK_DIR (4)            /* Convergence in orthogonality */
#define MP_MAXITER (5)           /* Maximum number of iterations reached */
#define MP_FTOL (6)              /* ftol is too small; no further improvement*/
#define MP_XTOL (7)              /* xtol is too small; no further improvement*/
#define MP_GTOL (8)              /* gtol is too small; no further improvement*/
#define MP_SIGINT (100)          /* PE: user typed Ctrl-C */

/* Double precision numeric constants */
#define MP_MACHEP0 2.2204460e-16
#define MP_DWARF   2.2250739e-308
#define MP_GIANT   1.7976931e+308

#if 0
/* Float precision */
#define MP_MACHEP0 1.19209e-07
#define MP_DWARF   1.17549e-38
#define MP_GIANT   3.40282e+38
#endif

#define MP_RDWARF  (sqrt(MP_DWARF*1.5)*10)
#define MP_RGIANT  (sqrt(MP_GIANT)*0.1)


/* External function prototype declarations */
extern int mpfit(mp_func funct, int m, int npar,
		 double *xall, mp_par *pars, mp_config *config, ModelObject *theModel, 
		 mp_result *result);


// Added by PE
void InterpretMpfitResult( int mpfitResult, std::string& interpretationString );

#endif /* _MPFIT_H_ */


