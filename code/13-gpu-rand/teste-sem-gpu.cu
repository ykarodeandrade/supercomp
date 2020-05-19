#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <iostream>

template <typename T>
double funcao_que_recebe_device_ou_host(T v) {
    thrust::fill(v.begin(), v.end(), 0.4);  
    double s = thrust::reduce(v.begin(), v.end(), 0.0, thrust::plus<double>());
    return s;
}

int main() {
    thrust::host_vector<double> vec(10);
    double s = funcao_que_recebe_device_ou_host(vec);
    std::cout << s << "\n";
}