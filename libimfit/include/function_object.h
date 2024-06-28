/*   Class interface definition for function_object.cpp [imfit]
 *
 * This is intended to be an abstract base class for the various
 * function objects (e.g., Sersic function, broken-exponential
 * function, etc.).
 * 
 */


// CLASS FunctionObject [base class]:

#ifndef _FUNCTION_OBJ_H_
#define _FUNCTION_OBJ_H_

#include <map>
#include <string>
#include <vector>

#include "psf_interpolators.h"

using namespace std;


/// Virtual base class for function objects (i.e., 2D image functions)
class FunctionObject
{
  public:
    // Constructors:
    FunctionObject( );

    // override in derived classes only if said class *does* have user-settable
    // extra parameters
    /// Boolean function: returns true if function can accept optional extra parameters
    /// (default = false)
    virtual bool HasExtraParams( ) { return(false); }
    /// Set optional extra parameters
    virtual int SetExtraParams( map<string, string>& )  { return -1; }
    /// Boolean function: returns true if optional extra parameters have been set
    virtual bool ExtraParamsSet( ) { return(extraParamsSet); }

    // override in derived classes only if said class is PointSource or similar
    /// Returns true if function models point sources (e.g., PointSource class)
    virtual bool IsPointSource( ) { return(false); };
    /// Tell point-source function about PSF image data
    virtual void AddPsfData( double *psfPixels, int nColumns_psf, int nRows_psf ) { ; };
    /// Pass in pointer to PsfInterpolator object (for point-source classes only)
    virtual void AddPsfInterpolator( PsfInterpolator *theInterpolator ) { ; };
    /// Returns string with name of interpolation type (point-source classes only)
    virtual string GetInterpolationType( ) { return string(""); };
    /// Sets internal oversamplingScale to specified value (for use in oversampled regions)
    virtual void SetOversamplingScale( int oversampleScale ) { ; };

    // probably no need to modify this (unless function uses subcomponent functions):
    virtual void SetSubsampling( bool subsampleFlag );

    // probably no need to modify this (for 1D functions):
    virtual void SetZeroPoint( double zeroPoint );

    // NEW MULTIMFIT STUFF
    // probably no need to modify this:
    virtual void SetImageParameters( double pixScale, double imageRot, double intensScale );

    virtual void AdjustParametersForImage( const double inputFunctionsParams[], 
										double adjustedFunctionParams[], int offsetIndex );


    // probably no need to modify this
    virtual void SetLabel( string & userLabel );

    // derived classes will almost certainly modify this, which
    // is used for pre-calculations and convolutions, if any:
    virtual void Setup( double params[], int offsetIndex, double xc, double yc );

    // all derived classes working with 1D data must override this:
    virtual void Setup( double params[], int offsetIndex, double xc );

    // all derived classes working with 2D images must override this:
    virtual double GetValue( double x, double y );

    // all derived classes working with 1D data must override this:
    virtual double GetValue( double x );

    // override in derived classes only if said class is a "background" object
    // which should *not* be used in total flux calculations
    /// Returns true if class can calculate total flux internally
    virtual bool IsBackground(  ) { return isBackground; }
    // override in derived classes only if said class *can* calcluate total flux
    /// Returns true if class can calculate total flux internally
    virtual bool CanCalculateTotalFlux(  ) { return(false); }
    // override in derived classes only if said class *can* calcluate total flux
    /// Returns total flux of image function, given most recent parameter values
    virtual double TotalFlux( ) { return -1.0; }

    // no need to modify this:
    virtual string GetDescription( );

    // no need to modify this:
    virtual string& GetShortName( );

    // no need to modify this:
    virtual string& GetLabel( );

    // probably no need to modify this:
    virtual void GetParameterNames( vector<string> &paramNameList );

    bool HasParameterUnits( ) { return parameterUnitsExist; };
    
    // probably no need to modify this:
    virtual void GetParameterUnits( vector<string> &paramUnitList );
    
    virtual void GetExtraParamsDescription( vector<string> &outputLines );

    // probably no need to modify this:
    virtual int GetNParams( );

    // Destructor (doesn't have to be modified, but MUST be declared
    // virtual in order for this to be a sensible base object
    // [see e.g. Scott Meyers, Effective C++]; otherwise behavior is 
    // undefined when a derived class is deleted)
    virtual ~FunctionObject();


  private:
  
  protected:
    int  nParams;  ///< number of input parameters that image-function uses
    bool  doSubsampling;
    bool  isBackground = false;
    bool  parameterUnitsExist = false;
    bool  extraParamsSet = false;
    vector<string>  parameterLabels, parameterUnits;
    map<string, string>  inputExtraParams;
    string  functionName, shortFunctionName, label;
    double  ZP;
    
    // NEW MULTIMFIT STUFF
    double  pixelScaling;    // multiply input size parameters by this
    double  intensityScale;  // multiply input intensity parameters by this
    double  imageRotation;   // treat image +y axis as rotated CCW relative to
                             // reference image by this many degrees

    // class member (constant char-vector string) which will hold name of
    // individual class in derived classes
    static const char  shortFuncName[];  ///< Class data member holding name of individual class
  
};

#endif   // _FUNCTION_OBJ_H_
