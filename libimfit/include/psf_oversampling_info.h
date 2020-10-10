// Header file for PsfOversamplingInfo class

#ifndef _PSF_OVERSAMPLING_INFO_H_
#define _PSF_OVERSAMPLING_INFO_H_

#include <string>
#include <tuple>

using namespace std;


// NOTE: the following class is used in PyImfit

class PsfOversamplingInfo
{
  public:
    PsfOversamplingInfo();
    PsfOversamplingInfo( double *inputPixels, int nCols, int nRows, int scale,
    					string inputRegionString, int xOffset=0, int yOffset=0,
    					bool isUnique=true, bool normalize=true );
    ~PsfOversamplingInfo();
  
    void AddPsfPixels( double *inputPixels, int nCols, int nRows, bool isUnique );
    void AddRegionString( string inputRegionString );
    void AddOversamplingScale( int scale );
    void AddImageOffset( int X0, int Y0 );
    void SetNormalizationFlag( bool normalize );
  
    int GetNColumns( );
    int GetNRows( );
    bool PixelsArrayIsUnique( );
    double * GetPsfPixels( );
    string GetRegionString( );
    int GetOversamplingScale( );
    void GetImageOffset( int &x0, int &y0 );
    std::tuple<int, int, int, int> GetCorrectedRegionCoords( );
    bool GetNormalizationFlag( );

  private:
    int  nColumns_psf, nRows_psf;
    int  X0_offset, Y0_offset;   // offset for starting coords within main image
    string  regionString;
    bool  pixelsArrayIsUnique;
    double *  psfPixels;
    int  oversamplingScale;
    bool  normalizePSF;
};


#endif   // _PSF_OVERSAMPLING_INFO_H_
