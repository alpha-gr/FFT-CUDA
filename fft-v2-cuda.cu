#include "fft-v2-cuda.cuh"
#include <thrust/complex.h>
#include <complex>
#include <math.h>
#include <vector>

#define FORWARD 1
#define REVERSE -1

bool MYpowerOf2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

__device__ void FFT1D(short dir, thrust::complex<double>* data, int length) {

    long n = 1 << length; // 2^length, numero di punti della FFT

    //applica il bit reversal al vettore
    int i2 = n >> 1; // n/2
    int k, j = 0;
    thrust::complex<double> tmp;
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
    thrust::complex<double> u, t;
    thrust::complex<double> c = {-1.0, 0.0};
    for (l=0;l<length;l++) {
        l1 = l2;
        l2 <<= 1;
		u = thrust::complex<double>(1.0, 0.0);
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

__global__ void FFT2D_GPU_RIGHE(thrust::complex<double>* data, thrust::complex<double>* temp, int n, int nlog2, short dir) {
    FFT1D(dir, data +(threadIdx.x*n), nlog2);
    for(int i = 0; i < n; i++) {
        temp[i*n + threadIdx.x] = data[threadIdx.x*n + i];
    }
}

__global__ void FFT2D_GPU_COLONNE(thrust::complex<double>* data, thrust::complex<double>* temp, int n, int nlog2, short dir) {
    FFT1D(dir, temp + (threadIdx.x*n), nlog2);
    for(int i = 0; i < n; i++) {
        data[i*n + threadIdx.x] = temp[threadIdx.x*n + i];
    }
  }

//bool _FFT2D(complex<double>** data, int nRighe, int nColonne, short dir) {
//    int i,j;
//    /* Transform the rows */
//    if (nRighe != nColonne || !MYpowerOf2(nRighe) || !MYpowerOf2(nColonne)) return(false);
//    int n = log2(nRighe);
//
//    complex<double>* temp1 = static_cast<complex<double>* >(malloc(nRighe * sizeof (complex<double>)));
//    for (j=0;j<nColonne;j++) {
//        for (i=0;i<nRighe;i++) {
//            temp1[i] = data[i][j];
//        }
//        FFT1D(dir,temp1, n);
//        for (i=0;i<nColonne;i++) {
//            data[i][j] = temp1[i];
//        }
//    }
//    free(temp1);
//
//    /* Transform the columns */
//    complex<double>* temp2 = static_cast<complex<double>* >(malloc(nColonne * sizeof (complex<double>)));
//    for (i=0;i<nRighe;i++) {
//        for (j=0;j<nColonne;j++) {
//            temp2[j] = data[i][j];
//        }
//        FFT1D(dir,temp2, n);
//        for (j=0;j<nRighe;j++) {
//            data[i][j] = temp2[j];
//        }
//    }
//    free(temp2);
//    return(true);
//}

bool FFT2D_GPU(std::complex<double>** data, int n, short dir) {

    int nlog2 = log2(n);

    // Alloca memoria per una matrice unidimensionale di thrust::complex<double>
    thrust::complex<double>* h_data = new thrust::complex<double>[n * n];

    // Copia i dati dalla matrice bidimensionale di std::complex<double> alla matrice unidimensionale di thrust::complex<double>
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_data[i * n + j] = thrust::complex<double>(data[i][j].real(), data[i][j].imag());
        }
    }
    

    thrust::complex<double>* data_gpu;
	thrust::complex<double>* temp_gpu;
    cudaMalloc((void**)&data_gpu, n * n * sizeof(thrust::complex<double>));
    cudaMalloc((void**)&temp_gpu, n * n * sizeof(thrust::complex<double>));
    cudaMemcpy(data_gpu, h_data, n * n * sizeof(thrust::complex<double>), cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(n);

    FFT2D_GPU_RIGHE <<<grid, block >>> (data_gpu, temp_gpu, n, nlog2, dir);
    cudaDeviceSynchronize();

    FFT2D_GPU_COLONNE <<<grid, block >>> (data_gpu, temp_gpu, n, nlog2, dir);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, data_gpu, n * n * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);
    cudaFree(data_gpu);
    cudaFree(temp_gpu);

    // converto nuovamente i dati
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			data[i][j] = std::complex<double>(h_data[i * n + j].real(), h_data[i * n + j].imag());
		}
	}

    return true;
}


void FFT_SHIFT(std::vector<std::vector<std::complex<double>>>& array, int rows, int cols) {
    // Shift delle righe
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols / 2; ++j) {
            std::swap(array[i][j], array[i][j + cols / 2]);
        }
    }
    // Shift delle colonne
    for (int i = 0; i < rows / 2; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::swap(array[i][j], array[i + rows / 2][j]);
        }
    }
}
