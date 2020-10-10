/*! \file
   \brief  Class declaration for Convolver (for convolving images with PSF).
  
 */

#ifndef _CONVOLVER_H_
#define _CONVOLVER_H_

#include <string>
#include <vector>

#include "fftw3.h"

using namespace std;


/// For debugging use: print a real-valued image to stdout
void PrintRealImage( double *image, int nColumns, int nRows );

/// For debugging use: print a complex-valued image to stdout
void PrintComplexImage_RealPart( fftw_complex *image_cmplx, int nColumns, int nRows );

/// For debugging use: print absolute value of complex-valued image to stdout
void PrintComplexImage_Absolute( fftw_complex *image_cmplx, int nColumns, int nRows );



// NOTE: The following class is used in PyImfit

/// Class for handling PSF convolution (stores PSF, performs convolutions 
/// with input model images)
class Convolver
{
  public:
    // Constructors and Destructors:
    Convolver( );
    ~Convolver( );
    
    // Public member functions:
    /// Set maximum number of FFTW threads
    void SetMaxThreads( int maximumThreadNumber );
    
    /// Supply PSF image to Convolver object
    void SetupPSF( double *psfPixels_input, int nColumns, int nRows,
    				bool normalize=true );
    
    void SetupImage( int nColumns, int nRows );
    
    /// Do final setup work (allocate things, generate FT of PSF image, etc.)
    int DoFullSetup( int debugLevel=0, bool doFFTWMeasure=false );

    /// Replace input model image (pixelVector) with convolution using stored PSF
    void ConvolveImage( double *pixelVector );


  private:
  // Private member functions:
  void ShiftAndWrapPSF( );
  
  // Data members:
  long  nPixels_image, nPixels_psf, nPixels_padded;
  int  nRows_psf, nColumns_psf;
  int  nRows_image, nColumns_image;
  int  nRows_padded, nColumns_padded;
  int  maxRequestedThreads;
  double  rescaleFactor;
  double  *psfPixels;
  double  *image_in_padded, *psf_in_padded, *convolvedImage_out;
  long  nPixels_padded_complex;
  fftw_complex  *image_fft_cmplx;
  fftw_complex  *psf_fft_cmplx;
  fftw_complex  *multiplied_cmplx;
  fftw_plan  plan_inputImage, plan_psf, plan_inverse;
  bool  psfInfoSet, imageInfoSet, fftVectorsAllocated, fftPlansCreated;
  bool  normalizePSF;
  int  debugStatus;
};


#endif  // _CONVOLVER_H_


// *** Things we do in convolution
// 
// [Should we work with complex array for model image within ModelObject, so that
// we don't repeatedly copy values from modelVector into modelImage_in
//    Note that mpfit works with the "deviates" array, which must be a simple array
// of doubles; 
//    -- Probably simpler to stick with modelVector in ModelObject the way it is
// (simple array of double), to keep the convolver interface simple (e.g., so that
// it can be used with 
// 
// 
// *** One-time setup for convolution:
// 
// [Subsample PSF via e.g. spline interpolation?]

  /* Normalize the PSF, if it's not already */

  /* Figure out how big the zero-padded images should be */
//  nRows_padded = nRows + nRows_psf - 1;
//  nColumns_padded = nColumns + nColumns_psf - 1;

  // Setup for FFT work:
//   modelImage_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nPixels_padded);
//   image_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nPixels_padded);
//   psf_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nPixels_padded);
//   psf_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nPixels_padded);
//   multiplied = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nPixels_padded);
//   convolvedModel = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nPixels_padded);
//   
//   plan_modelImage = fftw_plan_dft_2d(nColumns_padded, nRows_padded, modelImage_in, image_fft, FFTW_FORWARD,
//                              FFTW_ESTIMATE);
//   plan_psf = fftw_plan_dft_2d(nColumns_padded, nRows_padded, psf_in, psf_fft, FFTW_FORWARD,
//                              FFTW_ESTIMATE);
//   plan_inverse = fftw_plan_dft_2d(nColumns_padded, nRows_padded, multiplied, convolvedModel, FFTW_BACKWARD, 
//                              FFTW_ESTIMATE);

  // Populate (complex) psf array for FFT:

//   ShiftAndWrapPSF(psfPixels, nRows_psf, nColumns_psf, psf_in, nRows_padded, nColumns_padded);

  /* Do the forward FFTs for PSF: */
//   fftw_execute(plan_psf);


// *** Repeated image-convolution operations:

  // Populate (complex) input image array for FFT:
//   for (ii = 0; ii < nPixels_padded; ii++) {
//     modelImage_in[ii][0] = 0.0;
//     modelImage_in[ii][1] = 0.0;
//   }
//   for (ii = 0; ii < nRows; ii++)
//     for (jj = 0; jj < nColumns; jj++) {
//       modelImage_in[ii*nColumns_padded + jj][0] = allPixels[ii*nColumns + jj];
//     }
// 
// 
//   fftw_execute(plan_modelImage);

  /* Multiply the transformed arrays together: */

  /* Do the inverse FFT on the product array */

  /* Extract & rescale the real part of the convolved image */



