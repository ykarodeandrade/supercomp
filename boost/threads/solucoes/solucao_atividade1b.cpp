// Paralelize o código para calcular a soma mais rapido em vários cores
// Compilar: g++ atividade1.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt -lboost_timer-mt
// Referencia: https://theboostcpplibraries.com/boost.thread-management
// Resposta: Luciano Soares

#include <boost/timer/timer.hpp>
#include <iostream>
#include <cstdint>

#include <boost/thread.hpp>
#include <boost/chrono.hpp>

void thread(std::uint64_t *parcial, std::uint64_t inicio, std::uint64_t fim) {
    for (uint64_t i = inicio; i < fim; ++i)
        *parcial += i+1;
}

int main() {

    std::uint8_t n_th = boost::thread::hardware_concurrency();

    boost::timer::cpu_timer timer;

    std::uint64_t total = 0;

    std::uint64_t *parcial = new std::uint64_t[n_th*100];

    const std::uint64_t size = 2'147'483'647;
    boost::thread *t[n_th];
    for (std::uint64_t i = 0, j = 0; i < n_th; ++i, j+=size/n_th) {
        parcial[i*100]=0;
        t[i] = new boost::thread{thread,&parcial[i*100],j,(i+1==n_th?size:j+size/n_th)};
    }

    for (std::uint8_t i = 0; i < n_th; ++i)
        t[i]->join();

    for (std::uint8_t i = 0; i < n_th; ++i)
        total += parcial[i*100];

    std::cout << timer.format();
    std::cout << total << '\n';
}

// Ficou mais lento né, e agora? Tem uma dica nesse código para deixar mais rápido
// Estude o que é false sharing


