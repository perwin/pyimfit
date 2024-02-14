#ifndef _PSF_INTERPOLATORS_H_
#define _PSF_INTERPOLATORS_H_

// The following requires GSL version 2.0 or later
#include "gsl/gsl_spline2d.h"

#define kInterpolator_Base 0
#define kInterpolator_bicubic 1
#define kInterpolator_lanczos2 2
#define kInterpolator_lanczos3 3


// Auxiliary functions (public so we can test them)
double Lanczos( double x, int n );

int FindIndex( double xArray[], double xVal );



// Classes

class PsfInterpolator
{
  public:
  // need to provide zero-parameter base-class constructor, since derived-class
  // constructors will automatically call it
  PsfInterpolator( ) { ; };
  // derived classes should implement their own versions of this
  PsfInterpolator( double *inputImage, int nCols_image, int nRows_image ) { ; };
  
  virtual ~PsfInterpolator( ) { ; };
  
  int GetInterpolatorType( ) { return interpolatorType; };
  
  // pure virtual function (making this an abstract base class)
  virtual double GetValue( double x, double y ) = 0;

  protected:
    int  interpolatorType = kInterpolator_Base;
    // data members proper
    int  nColumns, nRows;
    long  nPixelsTot;
    double  xBound, yBound, deltaXMin, deltaXMax, deltaYMin, deltaYMax;
};


// Derived class using GNU Scientific Library's 2D bicubic interpolation
class PsfInterpolator_bicubic : public PsfInterpolator
{
  public:
  PsfInterpolator_bicubic( double *inputImage, int nCols_image, int nRows_image );
  
  ~PsfInterpolator_bicubic( );
  
  double GetValue( double x, double y );

  private:
    // new data members
    gsl_spline2d *splineInterp;
    gsl_interp_accel *xacc;
    gsl_interp_accel *yacc;
    double *xArray;
    double *yArray;
};


// Derived class using Lanczos2 kernel
class PsfInterpolator_lanczos2 : public PsfInterpolator
{
  public:
  PsfInterpolator_lanczos2( double *inputImage, int nCols_image, int nRows_image );
  
  ~PsfInterpolator_lanczos2( );
  
  double GetValue( double x, double y );

  private:
    // new data members
    double *xArray;
    double *yArray;
    double *psfDataArray;
};


// Derived class using Lanczos3 kernel
class PsfInterpolator_lanczos3 : public PsfInterpolator
{
  public:
  PsfInterpolator_lanczos3( double *inputImage, int nCols_image, int nRows_image );
  
~PsfInterpolator_lanczos3( );
  
  double GetValue( double x, double y );

  private:
    // new data members
    double *xArray;
    double *yArray;
    double *psfDataArray;
};

#endif   // _PSF_INTERPOLATORS_H_
