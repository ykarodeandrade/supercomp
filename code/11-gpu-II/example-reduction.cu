#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>

struct soma_impares {
    __device__ __host__
    int operator()(const double &x, const double &y) {
        return x + y;
    }
};

int main() {
    thrust::device_vector<double> v(100);
    thrust::sequence(v.begin(), v.end());

    double d = thrust::reduce(v.begin(), v.end(), 0.0, soma_impares());
    
    std::cout << v[99] << " " << d << "\n";

    return 0;
}