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
#define NUM_THREADS 4
static long num_steps = 100000000;
double step;
int main() {

	int i,nthreads;
	double x, pi, sum[NUM_THREADS];
    
	double start_time, run_time;

	step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
	start_time = omp_get_wtime();

    #pragma omp parallel
	{
		int i,id,n;
		double x;
		id = omp_get_thread_num();
		n = omp_get_num_threads();
	    if(id==0) nthreads=n;
        for(i=id,sum[id]=0.0;i<num_steps;i+=n){
			x=(i+0.5)*step;
			sum[id] += 4.0/(1.0+x*x);
        }
	}
	for(i=0,pi=0.0;i<nthreads;i++)
		pi += sum[i]*step;

	run_time = omp_get_wtime() - start_time;
	std::cout << "pi calculado com " << num_steps << " passos levou ";
	std::cout << run_time << " segundo(s) e chegou no valor : ";
	std::cout.precision(17);
	std::cout << pi << std::endl;
}