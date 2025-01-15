
#include <complex>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

bool FFT2D_GPU(std::complex<float>** data, int n, short dir);
void FFT_SHIFT(std::vector<std::vector<std::complex<float>>>& array, int rows, int cols);