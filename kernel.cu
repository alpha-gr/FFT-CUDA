
#pragma once
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
#include "fft-v2-cuda.cuh"

#define index(slice, ch, row, col, size, n_ch) (n_ch * size * size * slice) + (size * size * ch) + (size * row) + col


using namespace std;

int main() {

    cout << "Lettura del file..." << endl;

    string datafile = "C:/Users/user/source/repos/FFT/mridata/52c2fd53-d233-4444-8bfd-7c454240d314.h5";

    ISMRMRD::Dataset d(datafile.c_str(), "dataset", false);

    unsigned int num_acquisitions = d.getNumberOfAcquisitions();
    cout << "Number of acquisitions: " << num_acquisitions << endl;

    ISMRMRD::Acquisition acq;
    d.readAcquisition(0, acq);
    unsigned int num_channels = acq.active_channels();
    unsigned int num_samples = acq.number_of_samples();
    unsigned int num_slices = num_acquisitions / num_samples;

    // width and height of the slice

    cout << "Number of channels: " << num_channels << endl;
    cout << "Number of samples: " << num_samples << endl;
    cout << "Number of slices: " << num_slices << endl;

    // padded array size to perform FFT
    unsigned int size = next_power_of_two(num_samples);

    cout << "Reading data..." << endl;

    thrust::complex<float>* data;

	data = (thrust::complex<float>*)malloc(size * size * num_slices * num_channels * sizeof(thrust::complex<float>));

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
					data[index(slice, channel, row+pad, col+pad, size, num_channels)] = thrust::complex<float>(tmp.real(), tmp.imag());
                }
            }
        }
    
    }

	cout << "Processing data..." << endl;

    for (int slice = 0; slice < num_slices; slice++) {

        // 2D IFFT
        auto start = std::chrono::high_resolution_clock::now();
        for (int channel = 0; channel < num_channels; channel++) {

			FFT2D_GPU( data + index(slice, channel, 0, 0, size, num_channels), 512, 1);

            //FFT_SHIFT(slice_channels_padded[channel], padded_width, padded_height);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Tempo impiegato: " << duration_ms.count() << " millisecondi" << std::endl;


        // final vector to store the image
        vector<vector<float>> mri_image(size, vector<float>(size, 0.0));

        // combine the coils
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                float sumSquares = 0.0;
                for (int ch = 0; ch < num_channels; ++ch) {
                    // Magnitudine del valore complesso per il coil k
                    float magnitude = abs(data[index(slice,ch,row,col,size,num_channels)]);
                    sumSquares += magnitude * magnitude;
                }
                // Calcola il risultato RSS
                mri_image[row][col] = sqrt(sumSquares);
            }
        }


        // rotate the image by 90 degrees
        rotate_90_degrees(mri_image);

        // flip 
        //flipVertical(mri_image, padded_width, padded_height);
        //flipHorizontal(mri_image, padded_width, padded_height);

        string magnitudeFile = "C:/Users/user/source/repos/FFT-CUDA/output/" + to_string(slice) + ".png";

        write_to_png(mri_image, magnitudeFile);
    } // end for slice


    return 0;

}