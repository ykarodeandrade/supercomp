#include <iostream>
__global__ void mykernel(void) {
}
int main(void) {
	mykernel<<<1,1>>>();
	std::cout << "Hello World!\n";
	return 0;
}

