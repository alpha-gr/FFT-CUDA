
#include <complex>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/complex.h>

bool FFT2D_GPU(thrust::complex<float>* data, int size, int num_channels, int num_slices, short dir);
