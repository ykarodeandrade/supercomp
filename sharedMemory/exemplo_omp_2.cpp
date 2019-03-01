// Programa exemplo para mostrar funcionamento de sistema multithread.

#include <iostream>
#include <sstream>
#include <omp.h>

int main() {
	int acum=0;
    #pragma omp parallel
	{
		int n = omp_get_num_threads(); // armazenará o numero de thread
		int t = omp_get_thread_num(); // armazenará o identificador da thread
		acum+=1;
        std::stringstream s;
        s << "Thread " << t << "/" << n << ", acumulando " << acum << std::endl;
	    std::cout << s.str();
	}
	std::cout << "Valor final do acumulador " << acum << std::endl;
    return 0;
}
