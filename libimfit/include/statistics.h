/** @file
 * \brief code for computing statistics: mean, std.dev., confidence intervals, AIC, BIC.
 *
*/

#ifndef _STATISTICS_H_
#define _STATISTICS_H_

#include <tuple>


double Mean( double *vector, int nVals );

double StandardDeviation( double *vector, int nVals );

std::tuple<double, double> ConfidenceInterval( double *vector, int nVals );

// NOTE: the following two functions are used in PyImfit
double AIC_corrected( double logLikelihood, int nParams, long nData, int chiSquareUsed );

double BIC( double logLikelihood, int nParams, long nData, int chiSquareUsed );


#endif /* _STATISTICS_H_ */
