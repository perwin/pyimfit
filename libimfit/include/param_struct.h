/*    Definitions dealing with parameter structures, based on code originally
 * in Craig Markwardt's mpfit.h.
 *
 *    Currently, this is just the mp_par structure definition; we could conceivably
 * turn it into some kind of class definition in the future.
 *
 */

#include <string>

#ifndef _PARAM_STRUCT_H_
#define _PARAM_STRUCT_H_


// NOTE: (parts of) the following struct is used in PyImfit

/* Definition of a parameter constraint structure */
// This is a structure which holds metadata about a particular parameter:
// its name, whether or not it is fixed, any limits on its allowed values,
// etc.
struct mp_par_struct {
  int fixed;        /* 1 = fixed; 0 = free */
  int limited[2];   /* 1 = low/upper limit; 0 = no limit */
  double limits[2]; /* lower/upper limit boundary value */

  char *parname;    /* Name of parameter, or 0 for none */
  
  double offset;    /* Offset to be added when printing/writing output
                       (e.g., X0  or Y0 offset when an image subsection is
                       being fitted) */
  
  // The following are mpfit-related values and flags; we normally ignore
  // these, though mpfit itself will check their values (which we leave
  // set = 0.).
  double step;      /* Step size for finite difference */
  double relstep;   /* Relative step size for finite difference */
  int side;         /* Sidedness of finite difference derivative 
		        0 - one-sided derivative computed automatically
		        1 - one-sided derivative (f(x+h) - f(x)  )/h
		       -1 - one-sided derivative (f(x)   - f(x-h))/h
		        2 - two-sided derivative (f(x+h) - f(x-h))/(2*h) 
			3 - user-computed analytical derivatives
		    */
  int deriv_debug;  /* Derivative debug mode: 1 = Yes; 0 = No;

                       If yes, compute both analytical and numerical
                       derivatives and print them to the console for
                       comparison.

		       NOTE: when debugging, do *not* set side = 3,
		       but rather to the kind of numerical derivative
		       you want to compare the user-analytical one to
		       (0, 1, -1, or 2).
		    */
  double deriv_reltol; /* Relative tolerance for derivative debug
			  printout */
  double deriv_abstol; /* Absolute tolerance for derivative debug
			  printout */
};

typedef struct mp_par_struct mp_par;


struct SimpleParameterInfo {
  int fixed = 0;                 // 1 = fixed; 0 = free
  int limited[2] = {0,0};        // 1 = low/upper limit; 0 = no limit */
  double limits[2] = {0.0,0.0};  // lower/upper limit boundary value */

  std::string paramName = "";    // Name of parameter
  
  double offset = 0.0;           // Offset to be added when printing/writing output
                                 // (e.g., X0  or Y0 offset when an image subsection is
                                 // being fitted)
};


#endif /* _PARAM_STRUCT_H_ */
