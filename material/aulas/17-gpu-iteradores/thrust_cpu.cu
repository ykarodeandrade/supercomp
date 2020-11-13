#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

int main() {
    thrust::device_vector<double> aaa(10, 23);
    thrust::device_vector<double> bbb(10, 44);
    thrust::device_vector<double> res(10);

    thrust::transform(aaa.begin(), aaa.end(), bbb.begin(), res.begin(), thrust::plus<double>());


    for (auto el : res) {
        std::cout << el <<  " ";
    }

    std::cout << "\n";


    return 0;
}
