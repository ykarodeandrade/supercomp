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

int main() {
    std::cout << "Hello!\n";
    
    for (int i = 0; i < 10; i++) {
        auto vec = cria_vetor(1000);
        processa(vec, 1000);
        // vetor não é deletado no fim do main!
    }
    
    return 0;
}
