#include "fft-v2-cuda.cuh"
#include <thrust/complex.h>
#include <complex>
#include <math.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define rowIndex (num_channels * size * size * blockIdx.y) + (size * size * blockIdx.x) + (size * threadIdx.x)
#define sliceIndex (num_channels * size * size * blockIdx.y) + (size * size * blockIdx.x)

#define FORWARD 1
#define REVERSE -1

#define CHECK(call){\
    const cudaError_t error = call;\
    if (error != cudaSuccess) {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n", error,\
            cudaGetErrorString(error));\
    }\
}

__device__ void FFT1D(short dir, thrust::complex<float>* data, int length) {

    long n = 1 << length; // 2^length, numero di punti della FFT

    //applica il bit reversal al vettore
    int i2 = n >> 1; // n/2
    int k, j = 0;
    thrust::complex<float> tmp;
    for (int i=0; i<n-1; i++) {
        if (i < j) {
            tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
        k = i2;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j+=k;
    }

    //FFT (algoritmo cooley-tukey)
    int i, i1, l, l1, l2 = 1;
    thrust::complex<float> u, t;
    thrust::complex<float> c = {-1.0, 0.0};
    for (l=0;l<length;l++) {
        l1 = l2;
        l2 <<= 1;
		u = thrust::complex<float>(1.0, 0.0);
        for (j=0;j<l1;j++) {
            for (i=j;i<n;i+=l2) {
                i1 = i + l1;
                t = u * data[i1];
                data[i1] = data[i] - t;
                data[i] += t;
            }
            u = u * c;
        }

        c.imag(sqrt((1.0 - c.real()) / 2.0));
        if (dir == 1)
            c.imag(-c.imag());
        c.real(sqrt((1.0 + c.real()) / 2.0));
    }

    /* Scaling for forward transform */
    if (dir == 1) {
        for (i=0;i<n;i++) {
            data[i].real(data[i].real()/n);
            data[i].imag(data[i].imag()/n);
        }
    }

}

__global__ void FFT2D_GPU_COMPUTE(thrust::complex<float>* data, int size, int nlog2, int num_slices, int num_channels, short dir) {

    int i, j;
	// FFT delle righe
    FFT1D(dir, data + rowIndex, nlog2);

    __syncthreads();

    thrust::complex<float> tmp;
    // solo il primo thread scambia le righe con le colonne elemento per elemento
    if (threadIdx.x == 0) {
		for (i = 0; i < size; i++) {
			for (j = i; j < size; j++) {
                tmp = data[sliceIndex + i * size + j];
				data[sliceIndex + i * size + j] = data[sliceIndex + j * size + i];
				data[sliceIndex + j * size + i] = tmp;
			}
		}
    }
	__syncthreads();
    //FFT delle colonne
	FFT1D(dir, data + rowIndex, nlog2);

	__syncthreads();

    //shift delle righe e delle colonne
  //  if (threadIdx.x < size / 2) {
		//for (i = 0; i < size / 2; i++) {
		//	tmp = data[sliceIndex + (threadIdx.x * size) + i];
  //          data[sliceIndex + (threadIdx.x * size) + i] = data[sliceIndex + ((threadIdx.x + (size / 2)) * size) + i + (size / 2)];
  //          data[sliceIndex + ((threadIdx.x + (size / 2)) * size) + i + (size / 2)] = tmp;
		//}
  //  }
  //  else {
  //      for (i = 0; i < size/2; i++) {
		//	tmp = data[sliceIndex + (i * size) + threadIdx.x];
		//	data[sliceIndex + (i * size) + threadIdx.x] = data[sliceIndex + ((i + (size / 2)) * size) + threadIdx.x - (size / 2)];
		//	data[sliceIndex + ((i + (size / 2)) * size) + threadIdx.x - (size / 2)] = tmp;
  //      }
  //  }

    for (int i = 0; i < size / 2; i++) {
        //int dstRow = ((threadIdx.x + size / 2) % size);
        //int dstCol = ((i + size / 2) % size);

        tmp = data[sliceIndex + (threadIdx.x * size) + i];
        data[sliceIndex + (threadIdx.x * size) + i] =
            data[sliceIndex + (((threadIdx.x + size / 2) % size) * size) + ((i + size / 2) % size)];
        data[sliceIndex + (((threadIdx.x + size / 2) % size) * size) + ((i + size / 2) % size)] = tmp;
    }

    __syncthreads();
}

__global__ void FFT_SHIFT_GPU(thrust::complex<float>* data, thrust::complex<float>* temp, int n) {

	int n2 = n / 2;
    thrust::complex<float> tmp;
    // Shift delle righe
    for (int i = 0; i < n2; i++) {
		temp[threadIdx.x*n + i + n2] = data[threadIdx.x * n + i];
		temp[threadIdx.x * n + i] = data[threadIdx.x * n + i + n2];
    }

	// Barriera per sincronizzare i thread
    __syncthreads();

    // Shift delle colonne
    for (int i = 0; i < n2; i++) {
		data[i * n + threadIdx.x] = temp[(i + n2) * n + threadIdx.x];
		data[(i + n2) * n + threadIdx.x] = temp[i * n + threadIdx.x];
    }
}

bool FFT2D_GPU(thrust::complex<float>* data, int size, int num_channels, int num_slices, short dir) {

    int nlog2 = log2(size);
    unsigned int data_size = num_slices * num_channels * size * size * sizeof(thrust::complex<float>);

    thrust::complex<float>* data_gpu;
    CHECK(cudaMalloc((void**)&data_gpu, data_size));
	
    CHECK(cudaMemcpy(data_gpu, data, data_size, cudaMemcpyHostToDevice));

    dim3 grid(num_channels, num_slices);
    dim3 block(size);

    FFT2D_GPU_COMPUTE <<<grid, block>>> (data_gpu, size, nlog2, num_slices, num_channels, dir);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(data, data_gpu, data_size, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaFree(data_gpu));

    return true;
}
