#include <iostream>
#include <memory>
#include <vector>
#include "func.h"
#include "func.h"
#include "func.h"
#include "func.h"
#include "func.h"


int main() {
    std::cout << "Hello!\n";
    
    for (int i = 0; i < 10; i++) {
        auto vec = cria_vetor(1000);
        processa(vec, 1000);
        // vetor não é deletado no fim do main!
    }
    
    return 0;
}
