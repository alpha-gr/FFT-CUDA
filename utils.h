#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <complex>


using namespace std;

void write_to_png(vector<vector<double>> data, string outfile);
void combineCoils(const vector<vector<vector<complex<double>>>>& coils,
    vector<vector<double>>& image,
    int rows, int cols, int numCoils);
int next_power_of_two(int N);
void rotate_90_degrees(vector<vector<double>>& data);
vector<vector<complex<double>>> pad_vector(const vector<vector<complex<double>>>& data);
void apply_scale(std::vector<std::vector<double>>& magnitudes);
void flipVertical(std::vector<std::vector<double>>& image, int rows, int cols);
void flipHorizontal(std::vector<std::vector<double>>& image, int rows, int cols);