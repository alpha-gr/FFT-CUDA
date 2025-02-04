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
//double iStart = cpuSecond();
//double iElaps = cpuSecond() - iStart;

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

__device__ uint32_t reverse_bits_gpu(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

#define padded(x) ((x) + ((x)/WARP_SIZE)*SH_MEM_PADDING)
//#define padded(x) (x)
__global__ void kernel_fft(thrust::complex<float>* data) {


    uint32_t i = threadId_gpu;

    extern __shared__  thrust::complex<float> data_shared[];

    uint32_t rev = reverse_bits_gpu(i);
    rev = rev >> (32 - sizeLog2_gpu);
    data_shared[padded(rev)] = data[channelIndex_gpu + rowIndex_gpu + threadId_gpu];

    rev = reverse_bits_gpu(i+size2_gpu);
    rev = rev >> (32 - sizeLog2_gpu);
    data_shared[padded(rev)] = data[channelIndex_gpu + rowIndex_gpu + threadId_gpu + size2_gpu];

	__syncthreads();

    for (int s = 1; s <= sizeLog2_gpu; s++) {
        int mh = 1 << (s - 1);  // 2 ** (s - 1)

        // k = 2**s * (2*i // 2**(s-1))  for i=0..N/2-1
        // j = i % (2**(s - 1))  for i=0..N/2-1
        int k = threadIdx.x / mh * (1 << s);
        int j = threadIdx.x % mh;
        int kj = k + j;

        thrust::complex<float> a = data_shared[padded(kj)];

        float tr;
        float ti;

        // Compute the sine and cosine to find this thread's twiddle factor.
        sincosf(-(float)PI * j / mh, &ti, &tr);
        thrust::complex<float> twiddle = thrust::complex<float>(tr, ti);

        thrust::complex<float> b = twiddle * data_shared[padded(kj + mh)];

        // Set both halves of the array
        data_shared[padded(kj)] = a + b;
        data_shared[padded(kj + mh)] = a - b;

        __syncthreads();
    }
    data[channelIndex_gpu + rowIndex_gpu + threadId_gpu] = data_shared[padded(i)];
    data[channelIndex_gpu + rowIndex_gpu + threadId_gpu + size2_gpu] = data_shared[padded(i + size2_gpu)];
}

#define blockRow blockIdx.y
#define blockCol blockIdx.x
#define threadCol threadIdx.x
#define channelIndex_gpu (blockIdx.z * sizeSq_gpu)
__global__ void kernel_transpose(thrust::complex<float>* data) {
    if (blockRow > blockCol) return; //la metà bassa della diagonale non fa niente

    thrust::complex<float> temp;
    int i, j;

    if (blockRow == blockCol) {
        for (i = 0; i < TILE_DIM; i++) {
            if (threadCol > i) {
                temp = data[channelIndex_gpu + ((blockRow * TILE_DIM + i) * size_gpu) + (blockCol * TILE_DIM + threadCol)];
                data[channelIndex_gpu + ((blockRow * TILE_DIM + i) * size_gpu) + (blockCol * TILE_DIM + threadCol)] =
                    data[channelIndex_gpu + ((blockCol * TILE_DIM + threadCol) * size_gpu) + (blockRow * TILE_DIM + i)];
                data[channelIndex_gpu + ((blockCol * TILE_DIM + threadCol) * size_gpu) + (blockRow * TILE_DIM + i)] = temp;
            }
            else {
                break;
            }
        }
        return;
    }

    /*
     *  (BR*32+i)*size+(BC*32+TC) <-> (BC*32+TC)*size+(BR*32+i)
     */
    for (i = 0; i < 32; i++) {
        temp = data[channelIndex_gpu + ((blockRow * TILE_DIM + i) * size_gpu) + (blockCol * TILE_DIM + threadCol)];
        data[channelIndex_gpu + ((blockRow * TILE_DIM + i) * size_gpu) + (blockCol * TILE_DIM + threadCol)] =
            data[channelIndex_gpu + ((blockCol * TILE_DIM + threadCol) * size_gpu) + (blockRow * TILE_DIM + i)];
        data[channelIndex_gpu + ((blockCol * TILE_DIM + threadCol) * size_gpu) + (blockRow * TILE_DIM + i)] = temp;

    }

}
#undef blockRow
#undef blockCol
#undef threadCol
#undef channelIndex_gpu

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

    //num_slices = 16;

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

    dim3 grid_transpose(size/32, size/32, num_channels);
    dim3 block_transpose(32);

	int block_size = 32;
	dim3 grid_shift(size / 2 / block_size, size / block_size, num_channels);
	dim3 block_shift(block_size, block_size);

	dim3 grid_sum(size / block_size, size / block_size);
	dim3 block_sum(block_size, block_size);

    thrust::complex<float>* data_gpu;
    cudaMalloc((void**)&data_gpu, num_slices * num_channels * size * size * sizeof(thrust::complex<float>));


    for (int slice = 0; slice < num_slices; slice++) {
        cudaStreamCreate(&stream[slice]);

        cudaMemcpyAsync(data_gpu + sliceIndex, data + sliceIndex, num_channels * size * size * sizeof(thrust::complex<float>), cudaMemcpyHostToDevice, stream[slice]);
        
		kernel_fft << <grid, block, (size + (size/WARP_SIZE)*SH_MEM_PADDING) * sizeof(thrust::complex<float>), stream[slice] >> > (data_gpu + sliceIndex);
		kernel_transpose <<<grid_transpose, block_transpose, 32 * 32 * sizeof(thrust::complex<float>), stream[slice] >>> (data_gpu + sliceIndex);
        kernel_fft <<<grid, block, (size + (size / WARP_SIZE) * SH_MEM_PADDING) * sizeof(thrust::complex<float>), stream[slice] >>> (data_gpu + sliceIndex);
		//kernel_sum << <grid_sum, block_sum, 0, stream[slice] >> > (data_gpu + sliceIndex);
        kernel_freq_shift <<<grid_shift, block_shift, 0, stream[slice] >>> (data_gpu + sliceIndex);

        cudaMemcpyAsync(data + sliceIndex, data_gpu + sliceIndex, num_channels * size * size * sizeof(thrust::complex<float>), cudaMemcpyDeviceToHost, stream[slice]);
		cudaStreamDestroy(stream[slice]);
    }

    delete[] stream;

    cudaDeviceSynchronize();
    cudaFree(data_gpu);

    for (int slice = 0; slice < num_slices; slice++) {

        // final vector to store the image
        vector<vector<float>> mri_image(size, vector<float>(size, 0.0));

        // combine the coils
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                float sumSquares = 0.0;
                for (int ch = 0; ch < num_channels; ++ch) {

                    // Magnitudine del valore complesso per il coil k
                    float magnitude = abs(data[index(slice, ch, row, col, size, num_channels)]);
                    sumSquares += magnitude * magnitude;
                }
                // Calcola il risultato RSS
                mri_image[row][col] = sqrt(sumSquares);
                //if (col == 0) cout << sqrt(sumSquares) << endl;
            }
        }


        // rotate the image by 90 degrees
        //rotate_90_degrees(mri_image);

        // flip
        flipVertical(mri_image, size, size);
        flipHorizontal(mri_image, size, size);

        string magnitudeFile = argv[2] + to_string(slice) + ".png";

        write_to_png(mri_image, magnitudeFile);
    } // end for slice

    double iElaps = cpuSecond() - iStart;
	cout << "Elapsed time: " << iElaps << " s" << endl;


    return 0;

}