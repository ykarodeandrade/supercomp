// Questão 2:
// O código da q2 realiza um quicksort e funciona perfeitamente, contudo está sequencial
// Paralelize esse código de alguma forma eficiente usando um ambiente de memória compartilhada (OpenMP)
// Dica: se a quantidade de dados for muito pequena, começa a não ser interessante paralelizar.

#include <iostream>
#include <boost/random.hpp>
#include <omp.h>

#define SIZE  10000000

// escolhe um pivot, posiciona ele e divide o vetor em duas partes
unsigned int partition(std::vector<int>& vec, unsigned int low, unsigned int high) {
    
    int pivot = vec[high];          // escolhe um pivot 
    unsigned int index = (low - 1); // indice dos elementos de busca - 1

    for (unsigned int j = low; j<=high-1; j++) { 
        if (vec[j] <= pivot) {
            index++;
            std::swap(vec[index], vec[j]); 
        } 
    } 
    std::swap(vec[index+1], vec[high]); 
    return(index+1);
}

// Ordena os dados no vetor de low até high
void quicksort(std::vector<int>& vec, unsigned int low, unsigned int high)  { 

    if (low < high) {
        unsigned int pi = partition(vec, low, high); // particiona o vetor
        quicksort(vec, low, pi-1);
        quicksort(vec, pi+1, high);
    }

} 
  
// Mostra parte dos elementos do vetor
void printArray(std::vector<int>& vec) {

    unsigned int size = vec.size();
    unsigned int i;
    if(size<10) {
        for (i=0; i < size; i++) 
            std::cout << "#" << i << " : " << vec[i] << std::endl; 
    } else {
        for (i=0; i < 10; i++) {
            std::cout << "#" << i << " : " << vec[i] << std::endl;
        }
        if(size<20) {
            for (i=10; i < size; i++) 
                std::cout << "#" << i << " : " << vec[i] << std::endl; 
        } else {
            std::cout << "..." << std::endl; 
            for (i=size-10; i < size; i++) {
                std::cout << "#" << i << " : " << vec[i] << std::endl; 
            }
        }
    }

} 
  

int main() {

    std::vector<int> vec;

    // Gera uma série de número aleatórios
    boost::random::mt19937 gen{27};
    for(unsigned int i=0;i<SIZE;i++) {
        vec.push_back(gen());
    }

    double start_time, run_time;
    start_time = omp_get_wtime();
    
    quicksort(vec, 0, SIZE-1);  // processa o quicksort
    
    run_time = omp_get_wtime() - start_time;
    std::cout << "Tempo decorrido do quicksort: " << run_time << "s" << std::endl;

    std::cout << "Valores obtidos : " << std::endl;
    printArray(vec);

    return 0; 
} 



