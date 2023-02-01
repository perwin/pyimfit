// Code for determining the number of *physical* CPU cores on a computer;
// useful for determining how many threads to use for OpenMP and (especially)
// FFTW.

#ifndef _COUNT_CPU_CORES_H_
#define _COUNT_CPU_CORES_H_

int GetPhysicalCoreCount( );

#endif  // _COUNT_CPU_CORES_H_
