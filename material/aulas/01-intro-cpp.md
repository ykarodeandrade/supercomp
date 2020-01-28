O principal objetivo deste roteiro é retomar a prática de programação em C++ usando exercícios simples e que eventualmente sejam úteis para os projetos da disciplina. Os exercícios para entrega estarão indicados no fim do roteiro.

## Entrada e saída

Em C usamos as funções `printf` para mostrar dados no terminal e `scanf` para ler dados. Em C++ essas funções também podem ser usadas, mas em geral são substituídas pelos objetos `std::cin` e `std::cout` (disponíveis no cabeçalho iostream)

Para mostrar mensagens no terminal basta "enviar" dados para o objeto usando o operador <<. Veja o exemplo abaixo.

Em *C* usamos as funções `printf` para mostrar dados no terminal e `scanf` para ler dados. Em *C++* essas funções também podem ser usadas, mas em geral são substituídas pelos objetos `std::cin` e `std::cout` (disponíveis no cabeçalho `iostream`)

Para mostrar mensagens no terminal basta "enviar" dados para o objeto usando o operador `<<`. Veja o exemplo abaixo.

```cpp
int a = 10;
double b = 3.2;
std::cout << a << ";" << b << "\n";
```

Note que não precisamos mais usar a string de formatação cheia de `%d` e afins. Basta ir aplicando `<<` aos dados que queremos mostrar. 

O mesmo vale para a entrada, mas desta vez "tiramos" os dados do objeto `std::cin`. O exemplo abaixo lê um inteiro e um `double` do terminal. 

```cpp
int a;
double b;
std::cin >> a >> b;
```

!!! example
    Crie um programa que lê um número inteiro `n` e mostra em sua saída sua divisão fracionária por 2. Ou seja, antes de dividir converta `n` para `double`. 

## Alocação de memória

Em *C* usamos as funções `malloc` e `free` para alocar memória dinamicamente. Em *C++* essas funções também estão disponíveis, mas usá-las é considerado uma má prática. Ao invés, usamos os operadores `new` e `delete` para alocar memória. 

```cpp
point *p;
p = new point();

/* usar p aqui */

delete p;

```

Também podemos criar (e deletar) arrays de tamanho fixo usando `new[]` e `delete[]`. 

```cpp
int n;
std::cin >> n;
double *values = new double[n];

/* usar values aqui */

delete[] values;
```

!!! example 
    Crie um programa que lê um número inteiro `n` e depois lê `n` números fracionários $x_i$. Faça os seguintes cálculos e motre-os no terminal com 10 casas decimais. 

    $$\mu = \frac{1}{n} \sum_{i=1}^n x_i$$


    $$\sigma^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2$$

    **Dica**: procure por `setprecision` para configurar as casas decimais do `cout`.

!!! question short
    Você reconhece as fórmulas acima? Elas calculam quais medidas estatísticas?
    
## Contagem de tempo

Durante todo o curso iremos trabalhar com medições de tempo, especialmente para descobrir quais trechos do nosso código tem mais impacto no tempo de execução.

!!! example
    Crie um script python que gere uma entrada muito grande (`n=100000`) para o programa acima.

!!! question long
    Use a ferramenta `time` para medir o tempo de execução do programa. Escreva este valor abaixo. Você consegue dizer quanto tempo o cálculo da variância leva? 


Um dos problemas da utilização do comando `time` é que ele não separa o tempo gasto para ler a entrada do programa e o tempo gasto no calculo de cada medida. Felizmente a biblioteca padrão de *C++* possui diversas classes para medição de tempo.

O cabeçalho `<chrono>` disponibiliza diversas classes e funções para medição de tempo. Leia sua documentação [neste link](http://www.cplusplus.com/reference/chrono/).

!!! question long
    1. Qual classe você usaria para obter leituras de tempo com a melhor precisão possível? Quais métodos ou funções desta classe seriam úteis?
    2. Para que servem as classes `time_point` e `duration`?

!!! example
    Use as classes acima para medir o tempo de execução, separadamente, da média e da variância no exemplo anterior. Escreva abaixo o tempo gasto em *milisegundos*. 


## Projeto 0 - revisão de C++

A parte inicial de nosso curso foca na compreensão e implementação de funções matemáticas. Para isso vamos iniciar uma sequência de atividades que farão comparações de desempenho de funções do cabeçalho `cmath` (que é o mesmo `math.h` que usávamos em *C*, mas agora exportado para usar em *C++*). 

!!! warning
    O projeto 0 é individual e deverá ser entregue via blackboard. **Este projeto não deverá estar hospedado no github.** 


!!! example 
    Cria um arquivo chamado *parte0.c* contendo

    1. uma função `gera_vetor` que recebe um inteiro `n` e devolve um vetor de dados aleatório de tamanho `n` com tipo `double`. 
    1. funções `log`, `sqrt`, `pow3` (usando a biblioteca `math`) e `pow3mult` (usando o operador `*` duas vezes) que computam as operações correspondentes em cada elemento do vetor.
    1. uma função `sum` que calcula a soma do vetor 
    1. um `main` que cria vetores de tamanho incrementalmente maior e computa o tempo necessário para cada função rodar. 

!!! question long
    Para cada função criada, coloque abaixo os tempos colhidos para cada tamanho de vetor. 
    
!!! warning "O nome da disciplina é SuperComputação. Honre esse nome ao escolher tamanhos de vetores."
