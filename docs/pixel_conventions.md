# Pixel Coordinate Conventions

### Image Coordinates

Imfit was written to follow the standard 2D array indexing conventions of FITS, IRAF, and (e.g.) SAOimage
DS9, which are 1-based and column-major. This means that the center of the first pixel (in the lower-left 
of the image) has coordinates (x,y) = (1.0,1.0); the lower-left _corner_ of that pixel has coordinates (0.5,0.5),
and the upper-right corner of the same pixel is at (1.5,1.5). 
The first coordinate ("x") is the column number; the second ("y") is the row number.

To allow one to use Imfit configuration files with PyImfit, PyImfit adopts the same column-major, 1-based
indexing standard. The most obvious way this shows up is in the X0,Y0 coordinates
for the centers of function sets.

Python (and in particular NumPy), on the other hand, is 0-based and row-major. This means that the 
first pixel in the image is at (0,0); it also means that the first index is the _row_ number.

To translate coordinate systems, remember that a pixel with Imfit/PyImfit coordinates x,y 
would be found in a NumPy array at `array[y0 - 1,x0 - 1]`.


### Specifying Image Subsets for Fitting

The command-line version of Imfit allows you to specify image subsets for fitting
using a pixel-indexing convention similar to that of IRAF, which is 1-based and
inclusive of the limits -- i.e., you can fit a subset of the FITS file `data_image.fits`
using `data_image.fits[100:200,1203:1442]`, which will extract
and work on columns 100 through 200 and rows 1203 through 1442 of the image.

In a Python session (or script file or Jupyter notebook) you would instead simply apply the 
image-subset specification via indexing of NumPy arrays,
in which case you would necessarily be using Python indexing. So the previous
example would be done using (assuming you've read the `data_image.fits` file into
a NumPy variable named `data_image`) `data_image[1202:1442,99:200]`, since NumPy
arrays are row-major (y-values listed first) and use 0-based indexing, with the
upper index _excluded_.
