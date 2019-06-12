/** @file
 * \brief Class declaration for SolverResults, for storing results and messages from
 * minimization algorithms
 */
 
#ifndef _SOLVER_RESULTS_H_
#define _SOLVER_RESULTS_H_

#include <string>

#include "mpfit.h"

using namespace std;



/// \brief Class for storing useful results related to minimization.
class SolverResults 
{
  public:
    SolverResults( );
    ~SolverResults( );

    void AddMPResults( mp_result& mpResult );
    mp_result* GetMPResults( );

    void SetSolverType( int solverType );
    int GetSolverType( );

    void SetSolverName( string& name );
    string& GetSolverName( );

    void SetFitStatisticType( int fitStatType );
    int GetFitStatisticType( );
    
    void StoreInitialStatisticValue( double fitStatValue );
    double GetInitialStatisticValue( );

    void StoreBestfitStatisticValue( double fitStatValue );
    double GetBestfitStatisticValue( );

    void StoreNFunctionEvals( int nFunctionEvals );
    int GetNFunctionEvals( );
    
    bool ErrorsPresent( );
    void StoreErrors( double *errors, int nParams );
    void GetErrors( double *errors );


  private:
    int  whichSolver;
    int  whichFitStatistic;
    int  nParameters;
    int  nFuncEvals;
    double  initialFitStatistic;
    double  bestFitValue;
    bool  paramSigmasPresent;
    bool  paramSigmasAllocated;
    double  *paramSigmas;
    bool  mpResultsPresent;
    mp_result mpResult;
    string  solverName;
  
};

#endif   // _SOLVER_RESULTS_H_
