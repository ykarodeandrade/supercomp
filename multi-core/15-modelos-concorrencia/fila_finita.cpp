/* Código de exemplo para a disciplina SuperComputação 
 * 
 * Aviso: este não é um bom exemplo de práticas de programação em C++
 * nem uma boa implementação de fila. O código abaixo não funciona e é 
 * usado somente para explicar conceitos de programação concorrente. 
 * 
 * Autores: Igor Montagner, Luciano Soares
 */

#include <cassert>

template<typename T> class FilaFinita {
private:
    T *data;
    int to_consume, to_produce;
    int size;
    
public:
    FilaFinita(int s);
    ~FilaFinita();
    
    void add(T elem);
    T get();
};


template<typename T> FilaFinita<T>::FilaFinita(int s):
    size(s) {
    to_produce = 0; // espaço para colocar o próximo elemento
    to_consume = 0; // espaço contendo o próximo elemento a ser consumido
    data = new T[s];
}

template<typename T> FilaFinita<T>::~FilaFinita() {
    ~data;
}

template<typename T> void FilaFinita<T>::add(T elem) {
    data[to_produce] = elem;
    to_produce = (to_produce + 1) % size;
    assert(to_consume != to_produce);
}

template<typename T> T FilaFinita<T>::get() {
    assert(to_consume != to_produce);
    T elem = data[to_consume];
    to_consume = (to_consume + 1) % size;
    return elem;
}
