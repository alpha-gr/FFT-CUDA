
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
    unsigned int width = num_samples;
    unsigned int height = num_samples;

    cout << "Number of channels: " << num_channels << endl;
    cout << "Number of samples: " << num_samples << endl;
    cout << "Number of slices: " << num_slices << endl;

    cout << "width: " << width << endl;
    cout << "height: " << height << endl;

    // 3D array to store the multi channel slice data
    // num_channels x width x height
    vector<vector<vector<complex<float>>>> slice_channels(num_channels,
        vector<vector<complex<float>>>(width,
            vector<complex<float>>(height, { 0.0f, 0.0f })));

    // padded array to perform FFT
    unsigned int padded_width = next_power_of_two(width);
    unsigned int padded_height = next_power_of_two(height);

    vector<vector<vector<complex<float>>>> slice_channels_padded(num_channels,
        vector<vector<complex<float>>>(padded_width,
            vector<complex<float>>(padded_height, { 0.0f, 0.0f })));

    cout << "Processing data..." << endl;
    // Read the data from the acquisitions

    

    //num_slices = 10; // for testing purposes
    for (unsigned int slice = 0; slice < num_slices; slice++) {

        // Read the data for the current slice
        for (unsigned int j = 0; j < num_samples; j++) {
            d.readAcquisition(slice * num_samples + j, acq);
            for (unsigned int channel = 0; channel < num_channels; channel++) {
                for (unsigned int i = 0; i < num_samples; i++) {
                    slice_channels[channel][j][i] = acq.data(i, channel);
                }
            }
        }


        for (unsigned int channel = 0; channel < num_channels; channel++) {
            slice_channels_padded[channel] = pad_vector(slice_channels[channel]);
        }


        // 2D IFFT
        auto start = std::chrono::high_resolution_clock::now();
        for (unsigned int channel = 0; channel < num_channels; channel++) {

            // TODO rimuovere la costante 512
            complex<float>** data = new complex<float>*[512];

            for (size_t i = 0; i < 512; ++i) {
                data[i] = slice_channels_padded[channel][i].data();
            }

			FFT2D_GPU(data, 512, 1);

            //FFT_SHIFT(slice_channels_padded[channel], padded_width, padded_height);
			delete[] data;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Tempo impiegato: " << duration_ms.count() << " millisecondi" << std::endl;


        // final vector to store the image
        vector<vector<float>> mri_image(padded_width, vector<float>(padded_height, 0.0));

        // combine the coils
        combineCoils(slice_channels_padded, mri_image, padded_width, padded_height, num_channels);


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