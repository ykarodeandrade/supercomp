% Super Computação
% Igor Montagner, Luciano Soares
% 2019/2

# Introdução a C++

O principal objetivo deste roteiro é retomar a prática de programação em C++ usando exercícios simples e que eventualmente sejam úteis para os projetos da disciplina. Os exercícios para entrega estarão indicados no fim do roteiro. 

## Entrada e saída

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

### Exercício: 

Crie um programa que lê um número inteiro `n` e depois lê `n` números fracionários e calcula a média entre eles. 


## Alocação de memória

Em *C* usamos as funções `malloc` e `free` para alocar memória dinamicamente. Em *C++* essas funções também estão disponíveis, mas usá-las é considerado uma má prática. Ao invés, usamos os operadores `new` e `delete` para alocar memória. 

```cpp
point *p;
p = new point();

/* usar p aqui */

delete p;

```

Também podemos criar (e deletar) arrays de tamanho fixo usando `new[]` e `delete[]`. 
\newpage
```cpp
int n;
std::cin >> n;
double *values = new double[n];

/* usar values aqui */

delete[] values;
```

### Exercício

Crie um programa que receba um inteiro `n` e depois leia dois grupos de `n` números fracionários. Seu programa deverá imprimir a [distância euclidiana](https://en.wikipedia.org/wiki/Euclidean_distance) entre o primeiro e o segundo grupo de valores.


## Contagem de tempo

Durante todo o curso iremos trabalhar com medições de tempo.

### Exercício:

Crie um script python que gere uma entrada muito grande (`n=100000`) para o programa acima.

### Exercício: 

Use a ferramenta `time` para medir o tempo de execução do programa. Escreva este valor abaixo. Você consegue dizer quanto tempo o cálculo da distância euclidiana leva? 

\vspace{3em}


---- 

Um dos problemas da utilização do comando `time` é que ele não separa o tempo gasto para ler a entrada do programa e o tempo gasto no calculo da distância euclidiana. Felizmente a biblioteca padrão de *C++* possui diversas classes para medição de tempo.

### Exercício:

O cabeçalho `<chrono>` disponibiliza diversas classes e funções para medição de tempo. Leia sua documentação [neste link](http://www.cplusplus.com/reference/chrono/) e responda:


1. Qual classe você usaria para obter leituras de tempo com a melhor precisão possível? Quais métodos ou funções desta classe seriam úteis?
1. Para que servem as classes `time_point` e `duration`?

\vspace{7em}

### Exercício

Use as classes acima para medir o tempo de execução do cálculo da distância euclidiana no exemplo anterior. Escreva abaixo o tempo gasto em *milisegundos*. 

\newpage

## Exercício para entrega

Vamos iniciar nosso curso com uma comparação de desempenho no uso de números fracionários com precisão simples (`float`) ou dupla (`double`). Seu trabalho será:

1. criar uma função `gera_vetor` que recebe um inteiro `n` e devolve um vetor de dados aleatório de tamanho `n`. Você precisará de versões para `float` e `double`.
1. criar uma função `dist_euclid` que computa a distância euclidiana entre dois vetores. 
1. criar um `main` que cria vetores de tamanho incrementalmente maior e computa o tempo necessário para o cálculo da distância euclidiana entre dois vetores.

Com este programa pronto compare os desempenhos entre `float` e `double` e gere um gráfico. 


### Dicas:

1. Na aula comentamos sobre o tipo de variáveis `auto`. Como você pode usá-lo em seu programa para que não seja necessário criar várias cópias das funções acima.
1. Tente manter somente uma versão do código e utilizar macros `#define` e opções de compilação para criar executáveis diferentes a partir do mesmo código. 
1. O nome da disciplina é **Super**Computação. Honre esse nome ao escolher tamanhos de vetores. 

### Entrega:

Entregue seu código, um documento contendo os tempos para `double` e `float` e um gráfico que mostre as diferenças de tempo para cada tamanho de vetor.
