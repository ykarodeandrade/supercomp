// Gerador 10 números aleatórios
// Autor: Luciano P Soares

#include <boost/random/random_device.hpp>
#include <iostream>

int main() {
	boost::random::random_device gen;
	const long int MAX = gen.max();//0xFFFFFFFF 
	for(int f=0;f<10;f++) {
		std::cout << (double)gen()/MAX << '\n';	
	}
}