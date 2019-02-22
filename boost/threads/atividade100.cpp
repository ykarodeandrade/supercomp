// Paralelize o código para calcular a media mais rapido em vários cores
// Compilar: g++ atividade100.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt
// Autor: Luciano Soares <lpsoares@insper.edu.br>

#include <iostream>
#include <boost/timer.hpp>
#include <boost/random.hpp>

int main() {

    // cria sistema de números aleatórios em uma distribuição normal
    unsigned long seed = 89210;
    boost::mt19937 rng(seed);
    boost::normal_distribution <> norm(40.0,10.0);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution <>> normGen(rng,norm);

    // gera os números aleatórios em um vetor
    int numVars = 100000000;
    std::vector<uint8_t> myVec(numVars);
    double x;
    for(int i=0; i<numVars;i++) {
        x=normGen();
        myVec[i]=(uint8_t)(x<0?0:x);
    }

    // Inicia um timer (só para média)
    boost::timer t;

    unsigned long long int soma = 0;
    for(int i=0; i<numVars;i++) {
        soma += (int)myVec[i];
    }

    std::cout << soma/numVars << std::endl;

    // Exibe o tempo decorrido
    std::cout << "Tempo para calcular media:" << t.elapsed() << std::endl;

}