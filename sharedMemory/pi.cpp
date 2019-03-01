/*
Este programa calculará numericamente a integral de
                   4 / (1 + x * x)

de 0 a 1. O valor desta integral é pi - que
é ótimo, pois nos dá uma maneira fácil de verificar a resposta.
Esse programa Usa o temporizador do OpenMP
História: Escrito por Tim Mattson, 11/99.
          Atualizado por Luciano Soares
*/

#include <iostream>
#include <omp.h>

static long num_steps = 100000000;
double step;
int main() {

	int i;
	double x, pi, sum = 0.0;
	double start_time, run_time;

	step = 1.0 / (double)num_steps;

	start_time = omp_get_wtime();

	for (i = 0; i < num_steps; i++) {
		x = (i + 0.5) * step;
		sum = sum + 4.0 / (1.0 + x * x);
	}

	pi = step * sum;
	run_time = omp_get_wtime() - start_time;
	std::cout << "pi calculado com " << num_steps << " passos levou ";
	std::cout << run_time << " segundo(s) e chegou no valor : ";
	std::cout.precision(17);
	std::cout << pi << std::endl;
}