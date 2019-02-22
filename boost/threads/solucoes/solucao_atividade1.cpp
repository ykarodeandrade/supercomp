// Paralelize o código para calcular a soma mais rapido em vários cores
// Compilar: g++ atividade1.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt -lboost_timer-mt
// Referencia: https://theboostcpplibraries.com/boost.thread-management
// Resposta: Luciano Soares

#include <boost/timer/timer.hpp>
#include <iostream>
#include <cstdint>

#include <boost/thread.hpp>
#include <boost/chrono.hpp>

//#define NUM_THREADS 4

void thread(std::uint64_t *parcial, int inicio, int fim) {
    for (int i = inicio; i < fim; ++i)
        *parcial += i+1;
}

int main() {

    std::uint8_t n_th = boost::thread::hardware_concurrency();

    boost::timer::cpu_timer timer;

    std::uint64_t total = 0;

    std::uint64_t parcial[n_th][1];

    const int size = 2'147'483'647;
    boost::thread *t[n_th];
    for (int i = 0, j = 0; i < n_th; ++i, j+=size/n_th) {
        parcial[i][0]=0;
        t[i] = new boost::thread{thread,&parcial[i][0],j,(i+1==n_th?size:j+size/n_th)};
    }

    for (int i = 0; i < n_th; ++i)
        t[i]->join();

    for (int i = 0; i < n_th; ++i)
        total += parcial[i][0];

    std::cout << timer.format();
    std::cout << total << '\n';
}

// Ficou mais lento né, e agora? Tem uma dica nesse código para deixar mais rápido
// Estude o que é false sharing


