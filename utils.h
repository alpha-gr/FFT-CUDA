#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <complex>


using namespace std;

void write_to_png(vector<vector<float>> data, string outfile);
void combineCoils(const vector<vector<vector<complex<float>>>>& coils,
    vector<vector<float>>& image,
    int rows, int cols, int numCoils);
int next_power_of_two(int N);
void rotate_90_degrees(vector<vector<float>>& data);
vector<vector<complex<float>>> pad_vector(const vector<vector<complex<float>>>& data);
void apply_scale(std::vector<std::vector<float>>& magnitudes);
void flipVertical(std::vector<std::vector<float>>& image, int rows, int cols);
void flipHorizontal(std::vector<std::vector<float>>& image, int rows, int cols);