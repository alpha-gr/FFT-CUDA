
#include <complex>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

bool FFT2D_GPU(std::complex<double>** data, int n, short dir);
void FFT_SHIFT(std::vector<std::vector<std::complex<double>>>& array, int rows, int cols);