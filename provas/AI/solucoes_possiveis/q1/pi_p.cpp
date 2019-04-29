/*
Questão 1:
Peguei o código de calcular o pi usando o cálculo da integral de 4/1+xˆ2
Tentei paralelizar ele, mas acabei não sabendo que funções intrinsicas usar.
Ajuste esse código para que ele calcule o valor de pi de forma vetorizada.
*/

#include <iostream>
#include <x86intrin.h>
#include <chrono>

const long num_steps = 100000000;

int main() {

	double step, pi, sum = 0.0;

	step = 1.0 / (double)num_steps;

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

#ifdef SIMD

	__m256d indic, delta, x, x2, denominator, integral;
	__m256d vec = _mm256_setzero_pd();
	const __m256d vstep = _mm256_set1_pd(step);
	const __m256d one   = _mm256_set1_pd(1.0);
	const __m256d four  = _mm256_set1_pd(4.0);
	const __m256d incr  = _mm256_set_pd(0.5,1.5,2.5,3.5);

	for (int i = 0; i < num_steps; i+=4) {
		indic  = _mm256_set1_pd((double)i);
		//for(int j=0; j<4; ++j) ((double*)&indic)[j] = ((double)i);
		delta  = _mm256_add_pd(indic,incr);
		//for(int j=0; j<4; ++j) ((double*)&delta)[j] = ((double*)&indic)[j]+((double*)&incr)[j];
		x  = _mm256_mul_pd(delta,vstep);
		//for(int j=0; j<4; ++j) ((double*)&x)[j] = ((double*)&delta)[j]*((double*)&vstep)[j];
		x2 = _mm256_mul_pd(x,x);
		//for(int j=0; j<4; ++j) ((double*)&x2)[j] = ((double*)&x)[j]*((double*)&x)[j];
		denominator  = _mm256_add_pd(one,x2);
		//for(int j=0; j<4; ++j) ((double*)&denominator)[j] = ((double*)&one)[j]+((double*)&x2)[j];
		integral  = _mm256_div_pd(four,denominator);
		//for(int j=0; j<4; ++j) ((double*)&integral)[j] = ((double*)&four)[j]/((double*)&denominator)[j];
		vec = _mm256_add_pd(vec,integral);
		//for(int j=0; j<4; ++j) ((double*)&vec)[j] = ((double*)&vec)[j]+((double*)&integral)[j];
	}

	for(int i=0; i<4; ++i) sum += ((double*)&vec)[i]; // Não precisa alterar essa linha

#else

	double x;
	for (int i = 0; i < num_steps; i++) {
		x = (i + 0.5) * step;
		sum = sum + 4.0 / (1.0 + x * x);
	}

#endif

	pi = step * sum;
	std::cout << "pi calculado com " << num_steps << " passos em ";
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast< std::chrono::duration<double> >(end - start).count() << " segundos = ";
	std::cout.precision(17);
	std::cout << pi << std::endl;
}