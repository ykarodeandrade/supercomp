# 02 - Templates e STL

Esta prática consiste primariamente em aprender a usar tipos de dados complexos 
disponibilizados pela STL e pela biblioteca padrão de C++. A atividade principal desta prática 
consiste em modificar o exercício da aula passada para usar as estruturas de dados e 
funções da STL. 

## Smart Pointers

Uma das grandes dificuldades de desenvolver em C++ é evitar vazamentos de memória (memory leaks). Durante o desenvolvimento do programa é muito usado o recurso de alocação dinâmica de memória, contudo da mesma forma que o programador tem a responsabilidade de alocar a memória, ele tem de desalocar a memória. Os smart pointers são uma estratégia de evitar que você esqueça de desalocar e crie um programa devorador de memória. Os smart pointers percebem que uma memória alocada não é mais acessível e desaloca a memória.


**unique_ptr**: um smart pointer para um único objeto com um dono só. Ou seja, este smart pointer aponta para um objeto que deve ter só um apontamento de cada vez. Ao realizarmos atribuições a variável "dono" do objeto muda. 

**shared_ptr**: Um smart pointer para um único objeto e pode ter vários donos. Ou seja, este smart pointer aponta para um objeto que pode ter vários apontamentos de cada vez. Ao realizarmos atribuições adicionamos uma nova referência a este dado. Quando não existem mais referências o dado é automaticamente liberado usando `delete`

!!! example
    O programa abaixo (*tarefa1.cpp*) tem problemas de memória devido a alocação feita na função `cria_vetor` e não liberada a cada iteração do for. Conserte o programa usando `shared_ptr` para que a memória alocada por `cria_vetor` seja liberada automaticamente. 
    ```cpp
    #include <iostream>
    #include <memory>
    #include <vector>

    double *cria_vetor(int n) {
        double *ptr = new double[n];
        for (int i = 0; i < n; i++) {
            prt[i] = 0.0;
        }
        return ptr;
    }

    void processa(double *ptr, int n) {
        for (int i = 0; i < n; i++) {
            ptr[i] *= 2;
        }
    }

    int main() {
        std::cout << "Hello!\n";
        
        for (int i = 0; i < 10; i++) {
            double *vec = cria_vetor(1000);
            processa(vec, 1000);
            // vetor não é deletado no fim do main!
        }
        
        return 0;
    }
    ```

## Strings e Vector

Nesta seção iremos trabalhar com dois conteiners muito usados da STL: [string](http://www.cplusplus.com/reference/string/) e [vector](http://www.cplusplus.com/reference/vector/vector/). O objetivo é acostumá-los a consultar a documentação de C++ e entendê-la com autonomia. A STL tem uma quantidade enorme de recursos e aprender a pesquisar como usá-los é importante para sua proficiência. 

!!! example 
    Faça um programa que lê uma linha de texto (usando `std::getline`) e procure nela todas as ocorrências da palavra "hello". Você deverá implementar uma função

    `std::vector<int> find_all(std::string text, std::string term);` 

    que devolve um vetor com a posição de todas as ocorrências de `term` em `text`. Sua função `main` deverá mostrar os resultados da busca de maneira bem formatada. 

## Projeto 0 - adicionando STL

O código produzido na última aula parece muito com código *C* e usa muito pouco dos recursos introduzidos em *C++* para tornar nossos programas mais legíveis e fáceis de escrever. 

!!! example
    Modifique sua função `gera_vetor` para usar `std::vector` ao invés de arrays puros e para usar o cabeçalho `<random>`. A distribuição usada deverá ser uniforme real com limites 5 a 27.

!!! example
    Modifique todas as funções (`log`, `sqrt`, `pow3`, `pow3mult` e `sum`) para receber `std::vector`. Note que você não precisa mais receber como argumento o tamanho do vetor. 

!!! question short
    Qual deverá ser a assinatura das funções acima para evitar que ocorra cópia do `std::vector`?
    
!!! example 
    Continuando o exercício acima, use iteradores para percorrer seu `std::vector`. Para deixar o código mais legível use `auto`

Vamos agora trabalhar com programação funcional em *C++* para tornar nosso código menor: armazenamento de referências para funções em variáveis e definição de funções usando `lambda`. 

Podemos definir funções no meio de nosso programa usando a seguinte sintaxe:

```cpp
[&|=] (argumentos) -> retorno {
    corpo da função aqui
};
```

Uma função definida desta maneira pode usar as variáveis locais disponíveis no momento em que ela foi declarada (mesmo que não sejam passadas como argumento). A primeira parte da declaração define se essas variáveis serão copiadas `[=]` ou se uma referência para elas será utilizada na função `[&]`. O restante segue padrões normalmente usados em *C++*. 

Podemos inclusive, passar essas funções como argumentos e devolvê-las como resultado de funções. Para isto usamos o tipo `std::function` disponível no cabeçalho `functional`. Os exercícios abaixo foram extraídos do arquivo `exemplos-lambda.cpp`. Faça-os e cheque seus resultados rodando o programa. Se houver dúvida chame o professor. 

!!! question short
    Qual é o resultado do código abaixo?
    
    ```cpp
    int c = 2;
    std::function<double(int)> by_two = [=](int n) {
        return double(n) / c; 
    };
    std::cout << by_two(5) << "\n";
    ```

!!! question short
    Qual é o resultado do código abaixo?
    
    ```cpp
    int c = 2;
    std::function<double(int)> by_c = [&](int n) {
        return double(n) / c; 
    };
    std::cout << by_c(7) << "\n";
    c = 3;
    std::cout << by_c(7) << "\n";
    ```

!!! question
    Escreva abaixo o tipo de uma variável que guarda referência para as funções que testamos neste exercício
    
    * `log`
    * `sqrt`
    * `pow3`
    * `pow3mult`
    * `sum`
    
!!! example
    Crie uma função `std::vector<double> teste_incremental(`**tipo aqui**`)` que recebe um ponteiro para o tipo das funções acima e executa a função recebida com tamanhos de vetores incrementalmente maiores. Sua função deverá devolver os tempos (em segundos) para todas as execuções feitas. Ou seja, a função teste_incremental deverá funcionar como uma "casquinha" que gera vetores, roda as funções matemáticas testadas e retorna seus tempos de execução.

!!! example 
    você deve ter notado que a função `sum` não possui a mesma assinatura das outras. Use uma função `lambda` para adaptar os tipos e usar a função acima para testar a função `sum` também. 
