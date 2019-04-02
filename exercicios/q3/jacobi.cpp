#include <iostream> 
#include <cmath>
#include <ctime>
#include <chrono>
#include <omp.h>

#define MAX_ITERATIONS 1000
#define EPSILON 0.00000001

// O valor do elemento da diagonal tem de ser maior que a soma do resto da linha
bool is_dominant(float** A, int size){
	bool dominant = true;
	for (int i = 0; i < size; i++){
		float row_sum = 0;
		for (int j = 0; j < size; j++) {
			if (j != i) row_sum += std::abs(A[i][j]);
		}
		if (std::abs(A[i][i]) < row_sum) dominant=false;
	}
	return dominant;
}

void jacobi(float** A, int size, float* right_hand_side) {
	float* solution = new float[size];
	float* last_iteration = new float[size];

	for (int i = 0; i < size; i++) solution[i] = 0;

	for (int i = 0; i < MAX_ITERATIONS; i++){

		for (int i = 0; i < size; i++) {  // copia para recuperar valores
			last_iteration[i] = solution[i];
		}
		
		for (int j = 0; j < size; j++){
			float sigma_value = 0;
			for (int k = 0; k < size; k++){
				if (j != k) {
					sigma_value += A[j][k] * solution[k];
				}
			}
			solution[j] = (right_hand_side[j] - sigma_value) / A[j][j];
		}

		int stopping_count = 0;
		for (int s = 0; s < size; s++) {  // checando criterio de parada
			if (std::abs(last_iteration[s] - solution[s]) <= EPSILON) {
				stopping_count++;
			}
		}

		if (stopping_count == size) break;

#ifdef MANUAL
		std::cout << "Iteração " << i+1 << ": ";
		for (int l = 0; l < size; l++) {
			std::cout << solution[l] << " "; 
		}
		std::cout << std::endl;
#endif

	}
}

int main(){

	int size;
	float** A;
	float* b;

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

#ifdef MANUAL
	std::cout << "Digite o tamanho da matriz: ";
	std::cin >> size;
	
	// alocando as matrizes
	A = new float*[size];
	for (int i = 0; i < size; i++) A[i] = new float[size];
	b = new float[size];

	// Inserindo dados da matriz
	std::cout << "Digite os elementos da matriz:\n";
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++) {
			std::cout << "A[" << i << "][" << j << "]: ";
			std::cin >> A[i][j];
		}
	}
	
	if (!is_dominant(A, size)){
		std::cout << " A matriz não é diagonalmente dominante\n";
		return 1;
	}

	std::cout << "Digite os valores de b\n";
	for (int i = 0; i < size; i++){
		std::cout << "b[" << i << "]: ";
		std::cin >> b[i];
	}
#else

	size=30000;
	
	// alocando as matrizes
	A = new float*[size];
	for (int i = 0; i < size; i++) A[i] = new float[size];
	b = new float[size];

	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++) {
			if(i==j) A[i][j]=(size*(size-i)/0.6)+(size*i*0.7);
			else A[i][j]=(i/2.1)+(j/3.2);
		}
	}
	
	if (!is_dominant(A, size)){
		std::cout << " A matriz não é diagonalmente dominante\n";
		return 1;
	}

	for (int i = 0; i < size; i++){
		b[i]=i*2.1;
	}

#endif
	std::chrono::high_resolution_clock::time_point intermed = std::chrono::high_resolution_clock::now();
	std::cout << "Tempo para criação dos dados: ";
	std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(intermed - start).count() << " segundo.\n";
	
	jacobi(A, size, b);
	std::cout << "Tempo de processamento: ";
	std::cout << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - intermed).count() << " segundo.\n";
	return 0;
}



