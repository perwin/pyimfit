/** @file
    \brief Generally useful definitions & constants (debugging levels, fit statistics, solvers, etc.) 

    Definitions of constants referring to debugging levels, which fit statisitic
    is being used, which minimizer/solver is being used, format of error/weight
    image, definition of good/bad pixels in mask, and max buffer sizes for
    lines of text and for filenames.
 */

#ifndef _DEFINITIONS_H_
#define _DEFINITIONS_H_

#include <string>
#include <vector>
#include <map>
#include <signal.h>   // for sig_atomic_t

using namespace std;

#ifndef NO_SIGNALS
// volatile sig_atomic_t  stopSignal_flag;
extern volatile sig_atomic_t  stopSignal_flag;
#endif


const int  MAXLINE = 1024;
const int MAX_FILENAME_LENGTH = 512;


const double NO_MAGNITUDES = -10000.0;   /* indicates data are *not* in magnitudes */

const double GIGABYTE = 1073741824.0;   /* 1 gigabyte */
const double MEMORY_WARNING_LIMT = 1073741824.0;   /* 1 gigabyte */

// imfit-related
#define DEFAULT_IMFIT_CONFIG_FILE   "imfit_config.dat"
#define DEFAULT_OUTPUT_PARAMETER_FILE   "bestfit_parameters_imfit.dat"
#define USER_INTERRUPT_OUTPUT_PARAMETER_FILE   "current_parameters_imfit.dat"

// makeimage-related
#define DEFAULT_MAKEIMAGE_OUTPUT_FILENAME   "modelimage.fits"
const int DEFAULT_ESTIMATION_IMAGE_SIZE = 5000;


/* DEBUGGING LEVELS: */
const int  DEBUG_NONE  =             0;
const int  DEBUG_BASIC =             1;
const int  DEBUG_2     =             2;
const int  DEBUG_3     =             3;
const int  DEBUG_ALL   =            10;


/* OPTIONS FOR FIT STATISTICS: */
const int FITSTAT_CHISQUARE   =      1;   /// standard chi^2
const int FITSTAT_CASH        =      2;   /// standard (minimal) Cash statistic
const int FITSTAT_POISSON_MLR =      3;   /// Poisson Maximum Likelihood Ratio statistic
const int FITSTAT_CHISQUARE_DATA    =      10;   /// chi^2, per-pixel errors from data values
const int FITSTAT_CHISQUARE_MODEL   =      11;   /// chi^2, per-pixel errors from model values
const int FITSTAT_CHISQUARE_USER    =      12;   /// chi^2, per-pixel errors are user-supplied

const double DEFAULT_FTOL = 1.0e-8;



/* SOLVER OPTIONS: */
const int NO_FITTING           =     0;
const int MPFIT_SOLVER         =     1;
const int DIFF_EVOLN_SOLVER    =     2;
const int NMSIMPLEX_SOLVER     =     3;
const int ALT_SOLVER           =     4;
const int GENERIC_NLOPT_SOLVER =     5;

/* TYPE OF INPUT ERROR/WEIGHT IMAGE */
const int  WEIGHTS_ARE_SIGMAS    =  100;  /// "weight image" pixel value = sigma
const int  WEIGHTS_ARE_VARIANCES =  110;  /// "weight image" pixel value = variance (sigma^2)
const int  WEIGHTS_ARE_WEIGHTS   =  120;  /// "weight image" pixel value = weight

const int  MASK_ZERO_IS_GOOD =       10;  /// "standard" input mask format (good pixels = 0)
const int  MASK_ZERO_IS_BAD  =       20;  /// alternate input mask format (good pixels = 1)



/* STRING DEFINITIONS FOR PARAMETER NAMES */
const std::string  X0_string("X0");
const std::string  Y0_string("Y0");



/* Output string formats for printing parameter values */
#define XY_FORMAT "%s%s\t\t%.4f\n"
#define XY_FORMAT_WITH_ERRS "%s%s\t\t%.4f # +/- %.4f\n"
#define XY_FORMAT_WITH_LIMITS "%s%s\t\t%.4f\t\t%g,%g\n"
#define XY_FORMAT_WITH_FIXED "%s%s\t\t%.4f\t\tfixed\n"
// the following do *not* have "\n" at the end
#define PARAM_FORMAT "%s%s\t\t%7g"
#define PARAM_FORMAT_WITH_ERRS "%s%s\t\t%7g # +/- %.5g"
#define PARAM_FORMAT_WITH_LIMITS "%s%s\t\t%7g\t\t%g,%g"
#define PARAM_FORMAT_WITH_FIXED "%s%s\t\t%7g\t\tfixed"
#define UNITS_FORMAT "\t%s%s"

static vector< map<string, string> > EMPTY_MAP_VECTOR;

#endif /* _DEFINITIONS_H_ */
