/* Código de exemplo para a disciplina SuperComputação 
 * 
 * Aviso: este não é um bom exemplo de práticas de programação em C++
 * nem uma boa implementação de fila. O código abaixo não funciona e é 
 * usado somente para explicar conceitos de programação concorrente. 
 * 
 * Autores: Igor Montagner, Luciano Soares
 */

#include <cassert>

template<typename T> class FilaInfinita {
private:
    T *data;
    int to_consume, to_produce;
  
public:
    FilaInfinita();
    
    void add(T elem);
    T get();
};


template<typename T> FilaInfinita<T>::FilaInfinita() {
    to_produce = 0; // espaço para colocar o próximo elemento
    to_consume = 0; // espaço contendo o próximo elemento a ser consumido
    // supomos que data é magicamente alocado e contém espaço infinito
}

template<typename T> void FilaInfinita<T>::add(T elem) {
    data[to_produce] = elem;
    to_produce++;
}

template<typename T> T FilaInfinita<T>::get() {
    assert(to_consume < to_produce);
    T elem = data[to_consume];
    to_consume++;
    return elem;
}
