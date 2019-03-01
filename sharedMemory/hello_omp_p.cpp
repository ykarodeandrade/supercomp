// Hello world em C++

#include <iostream>
#include <omp.h>

int main() {
	#pragma omp parallel
    {
		int ID = omp_get_thread_num();
		std::cout << " Hello(" << ID << ")";
		std::cout << " World!(" << ID << ")\n";
	}
	return 0;
}