#include <iostream>
#include <chrono>
#include <thread>

static long num_steps = 1000000000;

void thread_pi(long start, long end, double *sum) {
    double sum_loc = 0;
    double step = 1.0 / (double)num_steps;
    for (long i = start; i < end; i++) {
        double x = (i + 0.5) * step;
        sum_loc = sum_loc + 4.0 / (1.0 + x * x);
    }
    *sum = sum_loc;
}

double step;
int main() {
    int nthreads = std::thread::hardware_concurrency();
    std::thread threads[nthreads];
    double *parcial = new double[nthreads];
    
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nthreads; i++) {
        long part_size = num_steps / nthreads + 1;
        long start = i * part_size;
        long end = start + part_size;
        threads[i] = std::thread(thread_pi, start, end, &parcial[i]);
    }
    
    double soma = 0;
    for (int i = 0; i < nthreads; i++) {
        threads[i].join();
        soma += parcial[i];
    }
    double step = 1.0 / (double)num_steps;
    double pi = soma * step; 
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::seconds> (end_time - start_time);
    
    std::cout << "O valor de pi calculado com " << num_steps << " passos levou ";
    std::cout << runtime.count() << " segundo(s) e chegou no valor : ";
    std::cout.precision(17);
    std::cout << pi << std::endl;
}
