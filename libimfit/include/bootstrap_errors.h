/*! \file
    \brief Public interfaces for function(s) dealing with estimating
    parameter errors via bootstrap resampling.

 */

#ifndef _BOOTSTRAP_ERRORS_H_
#define _BOOTSTRAP_ERRORS_H_

#include <string>

#include "param_struct.h"   // for mp_par structure
#include "model_object.h"


/*! \brief Primary wrapper function: runs bootstrap resampling and prints
           summary statistics; allows optional saving of all parameters to file

    If saving of all best-fit parameters to file is requested, then outputFile_ptr
    should be non-NULL (i.e., should point to a file object opened for writing, possibly
    with header information already written).
*/
int BootstrapErrors( const double *bestfitParams, vector<mp_par> parameterLimits, 
				const bool paramLimitsExist, ModelObject *theModel, const double ftol, 
				const int nIterations, const int nFreeParams, const int whichStatistic, 
				FILE *outputFile_ptr, unsigned long rngSeed=0, bool multimfitMode=false );


// NOTE: The following function is used in PyImfit

/*! \brief Alternate interface: returns array of best-fit parameters in outputParamArray,
           which is 1D contiguous array (not array of pointers to arrays); doesn't
           print any output at all or save anything to a file.
           
    Note that outputParamArray should be nParams*nIterations in size, and MUST BE 
    CORRECTLY ALLOCATED BY THE CALLER OF THIS FUNCTION! */
int BootstrapErrorsArrayOnly( const double *bestfitParams, vector<mp_par> parameterLimits, 
					const bool paramLimitsExist, ModelObject *theModel, const double ftol, 
					const int nIterations, const int nFreeParams, const int whichStatistic, 
					double *outputParamArray, unsigned long rngSeed=0, 
					bool verboseFlag=false );


#endif  // _BOOTSTRAP_ERRORS_H_
