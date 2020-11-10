#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>

struct raw_access {
    double *ptr;

    raw_access (double *ptr) : ptr(ptr) {};

    __device__ __host__
    double operator()(const int &i) {
        return ptr[i] + 1;
    }
};

int main() {
    thrust::device_vector<double> vec(10, 1);

    thrust::counting_iterator<int> iter(0);
    raw_access ra(thrust::raw_pointer_cast(vec.data()));
    thrust::transform(iter, iter+vec.size(), vec.begin(), ra);

    for (const double &d : vec) {
        std::cout << d << "\n";
    }

    return 0;
}
