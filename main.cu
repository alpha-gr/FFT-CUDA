#include "thrust/complex.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <chrono>
#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"
#include "ismrmrd/serialization.h"
#include "ismrmrd/xml.h"
#include "utils.h"
#include "lookup_tables.h"
#include <math.h>
#include <thread>

#define CHECK(call){\
    const cudaError_t error = call;\
    if (error != cudaSuccess) {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n", error,\
            cudaGetErrorString(error));\
    }\
}

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

#define PI 3.14159265358979323846

#define THREADS_PER_BLOCK 256
#define ROWS_PER_BLOCK 8
#define THREADS_PER_ROW (THREADS_PER_BLOCK/ROWS_PER_BLOCK)
#define SH_MEM_PADDING 1 // Shared memory padding to decrease bank conflicts
#define WARP_SIZE 32

#define index(slice, ch, row, col, size, n_ch) ((n_ch * size * size * slice) + (size * size * ch) + (size * row) + col)

#define sliceIndex (slice * num_channels * size * size)

#define sliceIndex_gpu (slice * numChannels_gpu * sizeSq_gpu)
#define channelIndex_gpu (channel_gpu * sizeSq_gpu)
#define rowGroupIndex_gpu (rowGroup_gpu * ROWS_PER_BLOCK * size_gpu)
#define rowIndex_gpu (rowId_gpu * size_gpu)
//#define index(slice, ch, row, col, size, n_ch) 1

#define channel_gpu blockIdx.y
#define rowGroup_gpu blockIdx.x
#define threadsPerRow_gpu = blockDim.x
#define rowId_gpu blockIdx.x
#define threadId_gpu threadIdx.x
#define TILE_DIM 32 //TRANSPOSE TILE DIMENSION IN PIXELS

__constant__ int numChannels_gpu;
__constant__ int size_gpu;
__constant__ int size2_gpu;
__constant__ int sizeLog2_gpu;
__constant__ int sizeSq_gpu;
__constant__ int samplesPerThread_gpu;
__constant__ int samplesPerThreadLog2_gpu;
__constant__ uint16_t bitrev_lookup_gpu[512];
__constant__ thrust::complex<float> W_gpu[10*512];

using namespace std;

__device__ uint32_t reverse_bits_gpu(uint32_t x, int sizeLog2)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    x = (x >> 16) | (x << 16);
	return x >> (32 - sizeLog2);
}

#define padded(x) ((x) + ((x)/WARP_SIZE)*SH_MEM_PADDING)
//#define padded(x) (x)
__global__ void kernel_fft(thrust::complex<float>* data) {

    uint32_t i = threadIdx.x;
	int idx = channelIndex_gpu + rowIndex_gpu + threadIdx.x;

    extern __shared__  thrust::complex<float> data_shared[];

    uint32_t rev1 = reverse_bits_gpu(i, sizeLog2_gpu);
    uint32_t rev2 = reverse_bits_gpu(i+size2_gpu, sizeLog2_gpu);

    data_shared[padded(rev1)] = data[idx];
    data_shared[padded(rev2)] = data[idx + size2_gpu];

	__syncthreads();

    for (int level = 1, step=1; level <= sizeLog2_gpu; level++, step*=2) {

        // k = 2**s * (2*i // 2**(s-1))  for i=0..N/2-1
        // j = i % (2**(s - 1))  for i=0..N/2-1
        int k = threadIdx.x / step * (1 << level);
        int j = threadIdx.x % step;
        int kj = k + j;

        thrust::complex<float> a = data_shared[padded(kj)];

        float tr;
        float ti;

        // Compute the sine and cosine to find this thread's twiddle factor.
        sincosf(-(float)PI * j / step, &ti, &tr);
        thrust::complex<float> twiddle = thrust::complex<float>(tr, ti);

        thrust::complex<float> b = twiddle * data_shared[padded(kj + step)];

        // Set both halves of the array
        data_shared[padded(kj)] = a + b;
        data_shared[padded(kj + step)] = a - b;

        __syncthreads();
    }
    data[idx] = data_shared[padded(i)];
    data[idx + size2_gpu] = data_shared[padded(i + size2_gpu)];
}

#define id threadIdx.x
#define miniBlockCol threadIdx.y
#define miniBlockRow threadIdx.z
#define blockCol blockIdx.x
#define blockRow blockIdx.y
#define channelIndex_gpu (blockIdx.z * sizeSq_gpu)
#define idMod4 (id & 3) //id % 4
#define idDiv4 (id >> 2) //id / 4
#define FULL_MASK 0xFFFFFFFF
#define MASK_0_TO_15 0x0000FFFF
__global__ void kernel_transpose(thrust::complex<float>* data) {

    const int lookup_table[32] = {
        16, 20, 24, 28, 17, 21, 25, 29,
        18, 22, 26, 30, 19, 23, 27, 31,
        0, 4, 8, 12, 1, 5, 9, 13,
        2, 6, 10, 14, 3, 7, 11, 15
    };

    int index;
    thrust::complex<float> tmp;
    if (blockRow > blockCol) return;
    if (blockRow == blockCol) {
        if (miniBlockRow > miniBlockCol) return;
        if (miniBlockRow == miniBlockCol) {
            if (id >= 16) return;
            index = channelIndex_gpu
                + (blockRow * 16 + miniBlockRow * 4 + idDiv4) * size_gpu
                + (blockCol * 16 + miniBlockCol * 4 + idMod4);

            tmp = data[index];
            tmp.real(__shfl_sync(MASK_0_TO_15, tmp.real(), lookup_table[id + 16], 16));
            tmp.imag(__shfl_sync(MASK_0_TO_15, tmp.imag(), lookup_table[id + 16], 16));
            data[index] = tmp;
            return;
        }
    }

    if (id < 16) {
        index = channelIndex_gpu
            + (blockRow * 16 + miniBlockRow * 4 + idDiv4) * size_gpu
            + (blockCol * 16 + miniBlockCol * 4 + idMod4);
    }
    else {
        index = channelIndex_gpu
            + (blockCol * 16 + miniBlockCol * 4 + idDiv4 - 4) * size_gpu
            + (blockRow * 16 + miniBlockRow * 4 + idMod4);
    }

    tmp = data[index];
    tmp.real(__shfl_sync(FULL_MASK, tmp.real(), lookup_table[id], 32));
    tmp.imag(__shfl_sync(FULL_MASK, tmp.imag(), lookup_table[id], 32));
    data[index] = tmp;

}
#undef id
#undef miniBlockCol
#undef miniBlockRow
#undef blockCol
#undef blockRow
#undef channelIndex_gpu
#undef idMod4
#undef idDiv4
#undef FULL_MASK
#undef MASK_0_TO_15

#define colId threadIdx.x
#define rowId blockIdx.x
__global__ void kernel_combineChannels(thrust::complex<float>* data, float* max) {

    extern __shared__ float shared_data[];

    //uchar4
    int baseIndex = blockIdx.x * size_gpu + threadIdx.x;
    float sum = 0, tmp;
    for (int ch = 0; ch < numChannels_gpu; ch++) {
        tmp = thrust::abs(data[baseIndex + ch * sizeSq_gpu]);
        sum += tmp * tmp;

    }
    shared_data[threadIdx.x] = sqrtf(sum);
    data[baseIndex].real(shared_data[threadIdx.x]);

    __syncthreads();

    //find max
    for (int stride = size2_gpu; stride > 0; stride /= 2) {
        if (threadIdx.x >= stride) return;
        tmp = shared_data[threadIdx.x + stride];
        if (shared_data[threadIdx.x] < tmp) {
            shared_data[threadIdx.x] = tmp;
        }
        __syncthreads();
    }
    max[blockIdx.x] = shared_data[0];
}
__global__ void kernel_max(float* data) {
    extern __shared__ float shared_data[];

    shared_data[threadIdx.x] = data[threadIdx.x];
    shared_data[threadIdx.x + size2_gpu] = data[threadIdx.x + size2_gpu];

    int stride = size2_gpu;
    float tmp;
    tmp = shared_data[threadIdx.x + stride];
    if (shared_data[threadIdx.x] < tmp) {
        shared_data[threadIdx.x] = tmp;
    }
    __syncthreads();
    for (stride /= 2; stride > 0; stride /= 2) {
        if (threadIdx.x >= stride) return;
        tmp = shared_data[threadIdx.x + stride];
        if (shared_data[threadIdx.x] < tmp) {
            shared_data[threadIdx.x] = tmp;
        }
        __syncthreads();
    }
    data[0] = shared_data[0];
}
__global__ void kernel_tochar(const thrust::complex<float>* __restrict__ data, const float* __restrict__ max, unsigned char* imgData) {
   
    imgData[(size_gpu -1 - blockIdx.x) * size_gpu + (size_gpu -1 - threadIdx.x)] = (unsigned char)((data[(blockIdx.x) * size_gpu + (threadIdx.x)].real() / *max) * 255);
}

__global__ void kernel_shiftToChar(const thrust::complex<float>* __restrict__ data, const float* __restrict__ max, unsigned char* imgData) {
    imgData[(size_gpu - 1 - ((rowId + size2_gpu) % size_gpu)) * size_gpu + (size_gpu - 1 - ((colId + size2_gpu) % size_gpu))] = (unsigned char)((data[rowId * size_gpu + colId].real() / *max) * 255);
}
#undef colId
#undef rowId

__global__ void kernel_freq_shift(thrust::complex<float>* data) {
	thrust::complex<float> tmp;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int new_row = (row + size2_gpu) % size_gpu;
	int new_col = (col + size2_gpu) % size_gpu;
	int channel = blockIdx.z;

	tmp = data[channel * sizeSq_gpu + row * size_gpu + col];
	data[ channel * sizeSq_gpu + row * size_gpu + col] = data[ channel * sizeSq_gpu + new_row * size_gpu + new_col];
	data[ channel * sizeSq_gpu + new_row * size_gpu + new_col] = tmp;

}

__global__ void kernel_sum(thrust::complex<float>* data) {
	float sum = 0.0;
	float magnitude = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int ch = 0; ch < numChannels_gpu; ch++) {
		magnitude = thrust::abs(data[ch * sizeSq_gpu + row * size_gpu + col]);
		sum += magnitude * magnitude;
	}
	data[row * size_gpu + col] = sqrt(sum);
}

int main(int argc, char* argv[]) {
    
    cout << "Lettura del file..." << endl;

    string datafile = argv[1];

    ISMRMRD::Dataset d(datafile.c_str(), "dataset", false);

    unsigned int num_acquisitions = d.getNumberOfAcquisitions();
    cout << "Number of acquisitions: " << num_acquisitions << endl;

    ISMRMRD::Acquisition acq;
    d.readAcquisition(0, acq);
    unsigned int num_channels = acq.active_channels();
    unsigned int num_samples = acq.number_of_samples();
    unsigned int num_slices = num_acquisitions / num_samples;

    // width and height of the slice

    num_slices = 16;

    cout << "Number of channels: " << num_channels << endl;
    cout << "Number of samples: " << num_samples << endl;
    cout << "Number of slices: " << num_slices << endl;

    // padded array to perform FFT
    unsigned int size = next_power_of_two(num_samples);
    int sizelog2 = log2(size);

    cout << "Loading data..." << endl;
    // Read the data from the acquisitions

    thrust::complex<float>* data;

    cudaMallocHost((void**)&data, num_slices*num_channels * size * size * sizeof(thrust::complex<float>));

	memset(data, 0, size * size * num_slices * num_channels * sizeof(thrust::complex<float>));

    //reading all the data with padding

	complex<float> tmp = complex<float>(0.0, 0.0);
	int pad = (size - num_samples) / 2;

    for (int slice = 0; slice < num_slices; slice++) {
        for (int row = 0; row < num_samples; row++) {
			d.readAcquisition(slice * num_samples + row, acq);
            for (int channel = 0; channel < num_channels; channel++) {
                for (int col = 0; col < num_samples; col++) {
                    tmp = acq.data(col, channel);
					data[index(slice, channel, (row+pad), (col+pad), size, num_channels)] = thrust::complex<float>(tmp.real(), tmp.imag());
                }
            }
        }

    }

    double iStart = cpuSecond();

    int constant_tmp;
    constant_tmp = num_channels;
    cudaMemcpyToSymbol(numChannels_gpu, &constant_tmp, sizeof(int));
    constant_tmp = size;
    cudaMemcpyToSymbol(size_gpu, &constant_tmp, sizeof(int));
    constant_tmp = size/2;
    cudaMemcpyToSymbol(size2_gpu, &constant_tmp, sizeof(int));
    constant_tmp = log2(size);
    cudaMemcpyToSymbol(sizeLog2_gpu, &constant_tmp, sizeof(int));
    constant_tmp = size*size;
    cudaMemcpyToSymbol(sizeSq_gpu, &constant_tmp, sizeof(int));
    constant_tmp = size/THREADS_PER_ROW;
    cudaMemcpyToSymbol(samplesPerThread_gpu, &constant_tmp, sizeof(int));
    constant_tmp = log2(size/THREADS_PER_ROW);
    cudaMemcpyToSymbol(samplesPerThreadLog2_gpu, &constant_tmp, sizeof(int));

    cudaStream_t* stream = new cudaStream_t[num_slices];
    dim3 grid(size, num_channels);
    dim3 block(size/2);

    dim3 grid_transpose(size / 4 / 4, size / 4 / 4, num_channels);
    dim3 block_transpose(32, 4, 4);;

	int block_size = 32;
	dim3 grid_shift(size / 2 / block_size, size / block_size, num_channels);
	dim3 block_shift(block_size, block_size);

	dim3 grid_sum(size / block_size, size / block_size);
	dim3 block_sum(block_size, block_size);

    thrust::complex<float>* data_gpu;
    cudaMalloc((void**)&data_gpu, num_slices * num_channels * size * size * sizeof(thrust::complex<float>));

    float* tmpMax_gpu;
    cudaMalloc((void**)&tmpMax_gpu, num_slices * size * sizeof(float));
    unsigned char* imgData_gpu;
    cudaMalloc((void**)&imgData_gpu, num_slices * size * size * sizeof(unsigned char));
    unsigned char* imgData;
    cudaMallocHost((void**)&imgData, num_slices * size * size * sizeof(unsigned char));

    for (int slice = 0; slice < num_slices; slice++) {
        cudaStreamCreate(&stream[slice]);

        cudaMemcpyAsync(data_gpu + sliceIndex, data + sliceIndex, num_channels * size * size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice, stream[slice]);
        
		kernel_fft << <grid, block, (size + (size/WARP_SIZE)*SH_MEM_PADDING) * sizeof(thrust::complex<float>), stream[slice] >> > (data_gpu + sliceIndex);
		kernel_transpose <<<grid_transpose, block_transpose,0, stream[slice] >>> (data_gpu + sliceIndex);
        kernel_fft <<<grid, block, (size + (size / WARP_SIZE) * SH_MEM_PADDING) * sizeof(thrust::complex<float>), stream[slice] >>> (data_gpu + sliceIndex);

        kernel_combineChannels << <size, size, size * sizeof(float), stream[slice] >> > (data_gpu + sliceIndex, tmpMax_gpu + slice * size);
        kernel_max << <1, size / 2, size * sizeof(float), stream[slice] >> > (tmpMax_gpu + slice * size);

        kernel_shiftToChar << <size, size, 0, stream[slice] >> > (data_gpu + sliceIndex, tmpMax_gpu + slice * size, imgData_gpu + slice * size * size);

        cudaMemcpyAsync(imgData + slice * size * size, imgData_gpu + slice * size * size, size * size * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream[slice]);

		cudaStreamDestroy(stream[slice]);
    }

    delete[] stream;

    cudaDeviceSynchronize();
    cudaFree(data_gpu);


    double iElaps = cpuSecond() - iStart;
    cout << "Elapsed time: " << iElaps << " s" << endl;
    
	vector<thread> threads;
    for (int slice = 0; slice < num_slices; slice++) {
        threads.emplace_back(writePNG, argv[2], slice, imgData + slice * size * size, size);
    }

    for (int slice = 0; slice < num_slices; slice++) {
        threads[slice].join();
    }

    return 0;

}