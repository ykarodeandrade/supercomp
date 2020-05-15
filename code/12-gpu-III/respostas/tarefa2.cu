#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <iostream>

struct adj_diff_positive {
    double *ptr;

    adj_diff_positive(double *ptr): ptr(ptr) {};

    double operator()(const int &i) {
        double diff = ptr[i+1] - ptr[i];
        return (diff > 0)?diff:0;
    }
};

struct is_diff_positive {
    double *ptr;

    is_diff_positive(double *ptr): ptr(ptr) {};

    bool operator()(const int &i) {
        double diff = ptr[i+1] - ptr[i];
        return diff > 0;
    }
};


int main() {
    thrust::host_vector<double> stocks_host;
    while (!std::cin.eof()) {
        double t;
        std::cin >> t;
        stocks_host.push_back(t);
    }

    thrust::device_vector<double>stocks_dev(stocks_host);
    
    is_diff_positive idp(thrust::raw_pointer_cast(stocks_dev.data()));
    int n = thrust::count_if(thrust::counting_iterator<int>(0),
                             thrust::counting_iterator<int>(stocks_dev.size()-1),
                             idp);
    
    adj_diff_positive adp(thrust::raw_pointer_cast(stocks_dev.data()));
    double s = thrust::transform_reduce(thrust::counting_iterator<int>(0),
                             thrust::counting_iterator<int>(stocks_dev.size()-1),
                             adp,
                             0.0,
                             thrust::plus<double>());

    std::cout << (s/n) << "\n";
    return 0;
}