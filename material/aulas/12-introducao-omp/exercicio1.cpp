#include <iostream>
#include <unistd.h>

double funcao1() {
    sleep(1);
    return 47;
}

double funcao2() {
    sleep(1);
    return -11.5;
}

int main() {
    double res_func1, res_func2;
    
    // TODO: chamar funcao1 e funcao2 em paralelo

    std::cout << res_func1 << " " << res_func2 << "\n";
}