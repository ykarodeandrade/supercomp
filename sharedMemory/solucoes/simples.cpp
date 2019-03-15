#include <iostream>

int main() {

	std::cout << "I think";
	
	#pragma omp parallel
	{
		#pragma omp master
		{
			#pragma omp task
			std::cout << " car";
			#pragma omp task
			std::cout << " race";
		}
	}

	std::cout << "s are fun\n";

}
