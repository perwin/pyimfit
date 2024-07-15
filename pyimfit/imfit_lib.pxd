# Cython header file for Imfit, by PE
# Copyright Peter Erwin, 2018--2024.

from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "definitions.h":
    int FITSTAT_CHISQUARE       =       1   # standard chi^2
    int FITSTAT_CASH            =       2   # standard (minimal) Cash statistic
    int FITSTAT_POISSON_MLR     =       3   # Poisson Maximum Likelihood Ratio statistic
    int FITSTAT_CHISQUARE_DATA  =      10   # chi^2, per-pixel errors from data values
    int FITSTAT_CHISQUARE_MODEL =      11   # chi^2, per-pixel errors from model values
    int FITSTAT_CHISQUARE_USER  =      12   # chi^2, per-pixel errors are user-supplied
    int WEIGHTS_ARE_SIGMAS    = 100 # "weight image" pixel value = sigma
    int WEIGHTS_ARE_VARIANCES = 110 # "weight image" pixel value = variance (sigma^2)
    int WEIGHTS_ARE_WEIGHTS   = 120 # "weight image" pixel value = weight
    int MASK_ZERO_IS_GOOD     =  10 # "standard" input mask format (good pixels = 0)
    int MASK_ZERO_IS_BAD      =  20 # alternate input mask format (good pixels = 1)
    int NO_FITTING           =     0
    int MPFIT_SOLVER         =     1
    int DIFF_EVOLN_SOLVER    =     2
    int NMSIMPLEX_SOLVER     =     3
    int ALT_SOLVER           =     4
    int GENERIC_NLOPT_SOLVER =     5


cdef extern from "statistics.h":
    double AIC_corrected( double logLikelihood, int nParams, long nData, int chiSquareUsed )
    double BIC( double logLikelihood, int nParams, long nData, int chiSquareUsed )


cdef extern from "param_struct.h":
    ctypedef struct mp_par:
        bint fixed          # 1 = fixed; 0 = free
        bint limited[2]     # 1 = low/upper limit; 0 = no limit
        double limits[2]    # lower/upper limit boundary value
 
        char *parname       # Name of parameter, or 0 for none
 
        double offset       # Offset to be added when printing/writing output
                            # (e.g., X0  or Y0 offset when an image subsection is
                            # being fitted)
        # note that there are additional fields within the struct which are
        # checked by mpfit(), but we always define those (in the C++ code) to be
        # 0, and they are not used elsewhere, so we don't include them in this
        # header.


# see get_images.cpp for how this object gets populated
# PsfOversamplingInfo::PsfOversamplingInfo( double *inputPixels, int nCols, int nRows,
# 										int scale, string inputRegionString,
# 										int xOffset, int yOffset, bool isUnique,
# 										bool normalize )

cdef extern from "psf_oversampling_info.h":
    cdef cppclass PsfOversamplingInfo:
        void AddPsfPixels( double *inputPixels, int nCols, int nRows, bool isUnique )
        void AddRegionString( string inputRegionString )
        void AddOversamplingScale( int scale )
        void AddImageOffset( int X0, int Y0 )
        void SetNormalizationFlag( bool normalize )
        int GetNColumns( )
        int GetNRows( )


cdef extern from "model_object.h":
    cdef cppclass ModelObject:
        # WARNING: calling SetupModelImage and AddImageDataVector in the
        # same ModelObject instance (or any of them more than once) will
        # cause memory leak!
        int SetupModelImage( int nImageColumns, int nImageRows )
        int AddImageDataVector( double *pixelVector, int nImageColumns, int nImageRows )
        void AddImageCharacteristics( double imageGain, double readoutNoise, double expTime,
                                int nCombinedImages, double originalSkyBackground )
        void AddErrorVector( long nDataValues, int nImageColumns, int nImageRows,
                         double *pixelVector, int inputType )
        int AddMaskVector( long nDataValues, int nImageColumns, int nImageRows,
                            double *pixelVector, int inputType )
        int AddPSFVector( long nPixels_psf, int nColumns_psf, int nRows_psf,
                            double *psfPixels, bool normalizePSF )
        int AddOversampledPsfInfo( PsfOversamplingInfo *oversampledPsfInfo )
        int UseModelErrors( )
        int UseCashStatistic( )
        void UsePoissonMLR( )
        int WhichFitStatistic( bool verbose=false )
        double GetFitStatistic( double *params )
        int FinalSetupForFitting( )
        void SetMaxThreads( int maxThreadNumber )
        int GetNParams( )
        int GetNFunctions( )
        long GetNDataValues( )
        long GetNValidPixels( )
        string& GetParameterName( int i )
        void CreateModelImage( double *params )
        double * GetModelImageVector( )
        double * GetWeightImageVector( )
        double FindTotalFluxes( double *params, int xSize, int ySize, double *individualFluxes )
        void SetVerboseLevel( int verbosity )
        void SetDebugLevel( int debuggingLevel )
        void SetOMPChunkSize( int chunkSize )

#         void PrintDescription()
#         void SetVerboseLevel(int level)


# int AddFunctions( ModelObject *theModel, const vector<string> &functionNameList,
#                   vector<string> &functionLabelList, vector<int> &functionSetIndices,
#                   const bool subamplingFlag, const int verboseFlag=0,
#                   vector< map<string, string> > &extraParams=EMPTY_MAP_VECTOR );

cdef extern from "add_functions.h":
    int GetFunctionParameterNames( string &functionName, vector[string] &parameterNameList );
    void GetFunctionNames( vector[string] &functionNameList )
    # Tricky thing: handling possible optional parameters for AddFunctions; these have
    # default values in .h file. so we need separate definitions for the three cases
    int AddFunctions( ModelObject *theModel, const vector[string] &functionNameList,
                  const vector[string] &functionLabelList, vector[int] &functionBlockIndices,
                  const bool subamplingFlag, const int verboseFlag,
                  vector[map[string, string]] &extraParams )
    int AddFunctions( ModelObject *theModel, const vector[string] &functionNameList,
                  const vector[string] &functionLabelList, vector[int] &functionBlockIndices,
                  const bool subamplingFlag, const int verboseFlag )
    int AddFunctions( ModelObject *theModel, const vector[string] &functionNameList,
                  const vector[string] &functionLabelList,
                  vector[int] &functionBlockIndices, const bool subamplingFlag )


cdef extern from "mpfit.h":
    ctypedef struct mp_result:
        double bestnorm
        double orignorm
        int niter
        int nfev
        int status
        int npar
        int nfree
        int npegged
        int nfunc
        # assuming we don't need the other fields in this struct ...


cdef extern from "solver_results.h":
    cdef cppclass SolverResults:
        int GetSolverType( )
        double GetBestfitStatisticValue( )
        mp_result* GetMPResults( )
        bool ErrorsPresent( )
        void GetErrors( double *errors )
        int GetNFunctionEvals()


cdef extern from "dispatch_solver.h":
    # two versions, to account for final optional parameter
    int DispatchToSolver( int solverID, int nParametersTot, int nFreeParameters,
                    int nPixelsTot, double *parameters, vector[mp_par] parameterInfo,
                    ModelObject *modelObj, double fracTolerance, bool paramLimitsExist,
                    int verboseLevel, SolverResults *solverResults, string& solverName,
                    unsigned long rngSeed )
    int DispatchToSolver( int solverID, int nParametersTot, int nFreeParameters,
                    int nPixelsTot, double *parameters, vector[mp_par] parameterInfo,
                    ModelObject *modelObj, double fracTolerance, bool paramLimitsExist,
                    int verboseLevel, SolverResults *solverResults, string& solverName )


cdef extern from "convolver.h":
    cdef cppclass Convolver:
        # two versions of SetupPSF, to account for final optional parameter
        void SetupPSF( double *psfPixels_input, int nColumns, int nRows )
        void SetupPSF( double *psfPixels_input, int nColumns, int nRows, bool normalize )
        void SetMaxThreads( int maximumThreadNumber )
        void SetupImage( int nColumns, int nRows )
        # three versions of SetupPSF, to account for default parameter values
        int DoFullSetup( int debugLevel, bool doFFTWMeasure )
        int DoFullSetup( int debugLevel )
        int DoFullSetup( )
        void ConvolveImage( double *pixelVector )


cdef extern from "bootstrap_errors.h":
    int BootstrapErrorsArrayOnly( const double *bestfitParams, vector[mp_par] parameterLimits,
                    const bool paramLimitsExist, ModelObject *theModel, const double ftol,
                    const int nIterations, const int nFreeParams, const int whichStatistic,
                    double *outputParamArray, unsigned long rngSeed, bool verboseFlag )

