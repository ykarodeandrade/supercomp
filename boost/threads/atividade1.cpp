// Paralelize o código para calcular a soma mais rapido em vários cores
// Compilar: g++ atividade1.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt -lboost_timer-mt
// Referencia: https://theboostcpplibraries.com/boost.thread-management

#include <boost/timer/timer.hpp>
#include <iostream>
#include <cstdint>

int main() {
    boost::timer::cpu_timer timer;

    const int size = 2'147'483'647;
    std::uint64_t total = 0;
    for (int i = 0; i < size; ++i)
        total += i+1;

    std::cout << timer.format();
    std::cout << total << '\n';
}