#include "utils.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void write_to_bmp(vector<vector<float>> data, string outfile) {

    char* filename = const_cast<char*>(outfile.c_str());
    int rows = data.size();
    int cols = data[0].size();

    // Crea un array di byte per l'immagine
    vector<unsigned char> imageData(rows * cols);

    // trova la massima magnitudine per normalizzare i valori
    float maxMagnitude = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (data[i][j] > maxMagnitude) {
                maxMagnitude = data[i][j];
            }
        }
    }

    // Normalizza le magnitudini e converte in byte
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            unsigned char pixelValue = static_cast<unsigned char>((data[i][j] / maxMagnitude) * 255);
            imageData[i * cols + j] = pixelValue;
        }
    }


    // Scrive il risultato come PNG
    //if (stbi_write_png(filename, cols, rows, 1, imageData.data(), cols) == 0) {
    //    cerr << "Errore nella scrittura dell'immagine in " << filename << endl;
    //    throw runtime_error("Errore nella scrittura dell'immagine");
    //}
    if (stbi_write_bmp(filename, cols, rows, 1, imageData.data()) == 0) {
        cerr << "Errore nella scrittura dell'immagine in " << filename << endl;
        throw runtime_error("Errore nella scrittura dell'immagine");
    }

    cout << "Immagine salvata come " << filename << endl;
}

void write_to_png(vector<vector<float>> data, string outfile) {

    char* filename = const_cast<char*>(outfile.c_str());
    int rows = data.size();
    int cols = data[0].size();

    // Crea un array di byte per l'immagine
    vector<unsigned char> imageData(rows * cols);

	// trova la massima magnitudine per normalizzare i valori
    float maxMagnitude = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (data[i][j] > maxMagnitude) {
                maxMagnitude = data[i][j];
            }
        }
    }

    // Normalizza le magnitudini e converte in byte
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            unsigned char pixelValue = static_cast<unsigned char>((data[i][j] / maxMagnitude) * 255);
            imageData[i * cols + j] = pixelValue;
        }
    }


    // Scrive il risultato come PNG
    if (stbi_write_png(filename, cols, rows, 1, imageData.data(), cols) == 0) {
        cerr << "Errore nella scrittura dell'immagine in " << filename << endl;
        throw runtime_error("Errore nella scrittura dell'immagine");
    }

    cout << "Immagine salvata come " << filename << endl;
}

// Funzione per combinare i risultati dei coil in una singola immagine con RSS (Root Sum of Squares)
void combineCoils(const vector<vector<vector<complex<float>>>>& coils,
    vector<vector<float>>& image,
    int rows, int cols, int numCoils) {
    // coils è un vettore 3D: [coil][row][col]
    // image è l'immagine combinata in output

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sumSquares = 0.0;
            for (int k = 0; k < numCoils; ++k) {
                // Magnitudine del valore complesso per il coil k
                float magnitude = abs(coils[k][i][j]);
                sumSquares += magnitude * magnitude;
            }
            // Calcola il risultato RSS
            image[i][j] = sqrt(sumSquares);
        }
    }
}

void writePNG(string outDir, int slice, unsigned char* imgData, int size) {
    if (stbi_write_png((outDir + to_string(slice) + ".png").c_str(), size, size, 1, imgData, size) == 0) {
        cerr << "Errore nella scrittura dell'immagine in " << outDir + to_string(slice) + ".png" << endl;
        throw runtime_error("Errore nella scrittura dell'immagine");
    }
    cout << "Immagine salvata come " << outDir + to_string(slice) + ".png" << endl;
}


int next_power_of_two(int N) {
    if (N <= 1) return 1; // Edge case for numbers <= 1
    int power = 1;
    while (power < N) {
        power <<= 1; // Equivalent to power = power * 2;
    }
    return power;
}

void rotate_90_degrees(vector<vector<float>>& data) {
    int n = data.size();
    for (int i = 0; i < n / 2; i++) {
        for (int j = i; j < n - i - 1; j++) {
            float temp = data[i][j];
            data[i][j] = data[n - j - 1][i];
            data[n - j - 1][i] = data[n - i - 1][n - j - 1];
            data[n - i - 1][n - j - 1] = data[j][n - i - 1];
            data[j][n - i - 1] = temp;
        }
    }
}

vector<vector<complex<float>>> pad_vector(const vector<vector<complex<float>>>& data) {
    int oldSize = data.size();
    int newSize = next_power_of_two(oldSize);
    int pad = (newSize - oldSize) / 2;

    vector<vector<complex<float>>> padded(newSize, vector<complex<float>>(newSize, { 0.0, 0.0 }));

    for (int i = 0; i < oldSize; ++i) {
        for (int j = 0; j < oldSize; ++j) {
            padded[i + pad][j + pad] = data[i][j];
        }
    }

    return padded;
}

// Funzione per applicare la scala logaritmica ai valori di magnitudine
void apply_scale(std::vector<std::vector<float>>& magnitudes) {
    for (auto& row : magnitudes) {
        for (auto& value : row) {
            value = sqrt(1 + value); // Applica la scala logaritmica
        }
    }
}

void flipVertical(std::vector<std::vector<float>>& image, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols / 2; ++j) {
            // Swap pixels across the vertical axis
            std::swap(image[i][j], image[i][cols - j - 1]);
        }
    }
}

void flipHorizontal(std::vector<std::vector<float>>& image, int rows, int cols) {
	for (int i = 0; i < rows / 2; ++i) {
		for (int j = 0; j < cols; ++j) {
			// Swap pixels across the horizontal axis
			std::swap(image[i][j], image[rows - i - 1][j]);
		}
	}
}