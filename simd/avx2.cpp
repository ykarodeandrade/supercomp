// Baseado em: Matt Scarpino
// Luciano Soares

#include <x86intrin.h>
#include <iostream>

int main() {

	float int_array[8] = {100, 200, 300, 400, 500, 600, 700, 800};

	// Criando vetor de máscara
	__m256i mask = _mm256_setr_epi32(-20, -72, -48, -9, -100, 3, 5, 8);

	// Carregando dados de forma seletiva no vetor
	__m256 result = _mm256_maskload_ps(int_array, mask);

	// Exibindo o resultado da operação
	float* res = (float*)&result;
	for(int i=0; i<8; ++i) std::cout << res[i] << ' ';
	std::cout << std::endl;

}