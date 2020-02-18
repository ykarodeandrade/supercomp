#include <iostream>
#include <memory>
#include <vector>


std::shared_ptr<double[]> cria_vetor(int n) {
    std::shared_ptr<double[]> prt(new double[n]);
    for (int i = 0; i < n; i++) {
        prt[i] = 0.0;
    }
    return prt;
}

void processa(std::shared_ptr<double[]>ptr, int n) {
    for (int i = 0; i < n; i++) {
        ptr[i] *= 2;
    }
}

