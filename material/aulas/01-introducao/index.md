# 01 - Recursos úteis de C++

A disciplina utilizará a linguagem C++ para implementação dos programas. Ela é muito usada em implementações de alto desempenho e possui recursos muito úteis e que simplificam a programação se comparada com C puro. Nas aulas 01 e 02 aprenderemos alguns desses recursos e os utilizaremos para implementação de algoritmos simples. 

!!! failure "Gabaritos e respostas"
    Este curso não fornece código de resposta para os exercícios de sala. Cada exercício é acompanhado de um algoritmo em pseudo-código e alguns pares de arquivos entrada/saída. Isto já é suficiente para que vocês verifiquem se sua solução está correta. 

    Boas práticas de programação serão demonstradas em exercícios corrigidos pelo professor durante o semestre. Uma lista (não exaustiva) dessas práticas estará disponíveis na página [Projeto](/projetos/checklist.md).

## Compilação 

Programas em C++ são compilados com o comando `g++`. Ele funciona igual ao `gcc` que vocês já usaram em Desafios e Sistemas Hardware-Software.

```
$> g++ arquivo.cpp -o executavel
```

## Entrada e saída em C++

Em C usamos as funções `printf` para mostrar dados no terminal e `scanf` para ler dados. Em C++ essas funções também podem ser usadas, mas em geral são substituídas pelos objetos `std::cin` e `std::cout` (disponíveis no cabeçalho iostream). 

A maior vantagem de usar `cin` e `cout` é que não precisamos mais daquelas strings de formatação estranhas com `%d`, `%s` e afins. Podemos passar variáveis diretamente para a saída do terminal usando o operador `<<`. Veja um exemplo abaixo. 

```cpp
int a = 10;
double b = 3.2;
std::cout << "Saída: " << a << ";" << b << "\n";
```

!!! example 
    Crie um arquivo `entrada-saida.cpp` com uma função `main` que roda o código acima. Compile e execute seu programa e verifique que ele mostra o valor correto no terminal. 

O mesmo vale para a entrada, mas desta vez "tiramos" os dados do objeto `std::cin`. O exemplo abaixo lê um inteiro e um `double` do terminal. 

```cpp
int a;
double b;
std::cin >> a >> b;
```

!!! example
    Modifique seu programa `entrada-saida.cpp` para ler ê um número inteiro `n` e mostrar sua divisão fracionária por 2. Ou seja, antes de dividir converta `n` para `double`. 


!!! hint "E esse `std::`?"
    Em `C++` podemos ter várias funções, variáveis e objetos em geral com o mesmo nome. Para evitar que eles colidam e não se saiba a qual estamos nos referindo cada nome deve ser definido um `namespace` (literalmente *espaco de nomes*). Podemos ter `namespace`s aninhados.Por exemplo, `std::chrono` contém as funções relacionadas contagem de tempo durante a execução de um programa. 

    Todas as funções, classes e globais na biblioteca padrão estão definidas no espaço `std`. Se quisermos, podemos omitir escrever `std::` toda vez digitando `using namespace std`. Isso pode ser feito também com namespaces aninhados. 

A implementação de algoritmos definidos usando expressões matemáticas é uma habilidade importante neste curso.

!!! example
    Escreva um programa que receba um inteiro `n` e calcule a seguinte série.

    $$
    S = \sum_{i=0}^n \frac{1}{2^i}
    $$

    Mostre as primeiras 15 casas decimais de `S`. Veja a documentação de [`std::setprecision` aqui](http://cplusplus.com/reference/iomanip/setprecision/). 

    ??? details "Resposta"
        Essa série converge para o número 2, mas sua resposta deverá ser sempre menor que este número. Logo, quanto maior `n` mais próxima sua resposta será. Seu programa deverá implementar algo como o algoritmo abaixo.

        ```
        leia inteiro n
        s = 0.0
        para i=0 até n
            s += 1 / (2 elevado a i)
        
        print(s)
        ```

## Alocação de memória

Em *C* usamos as funções `malloc` e `free` para alocar memória dinamicamente. Um inconveniente dessas funções é que sempre temos que passar o tamanho que queremos em bytes. Em *C++* essas funções também estão disponíveis, mas usá-las é considerado uma má prática. Ao invés, usamos os operadores `new` e `delete` para alocar memória. Existem duas vantagens em usá-los.

1. Podemos escrever diretamente o tipo que queremos, em vez de seu tamanho em bytes. 
2. A alocação de arrays é feita de maneira natural usando os colchetes `[]`.

Vejamos o exemplo abaixo. 

```cpp
int n;
std::cin >> n;
double *values = new double[n];

/* usar values aqui */

delete[] values;
```

É alocado um vetor de `double` de tamanho `n` (lido do terminal). Após ele ser usado liberamos o espaço alocado usando `delete[]`. 

!!! tip "E se eu quiser alocar um só valor?"
    É simples! É só usar `new` sem os colchetes `[]`!

!!! example 
    Crie um programa que lê um número inteiro `n` e depois lê `n` números fracionários $x_i$. Faça os seguintes cálculos e motre-os no terminal com 10 casas decimais. 

    $$\mu = \frac{1}{n} \sum_{i=1}^n x_i$$


    $$\sigma^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2$$

    ??? details "Resposta" 
         Os arquivos *t4-in-(1,2,3).txt* e *t4-out-(1,2,3).txt* devem ser usados para testar seu programa. 


!!! question short
    Você reconhece as fórmulas acima? Elas calculam quais medidas estatísticas?

    ??? details "Resposta"
        Média e variância.

## Vetores em C++

Apesar do uso de `new[]` e `delete[]` mostrado na seção anterior já ser mais conveniente, ainda são essencialmente um programa em C com sintaxe ligeiramente mais agradável. Para tornar a programação em C++ mais produtiva sua biblioteca padrão conta com estruturas de dados prontas para uso. 

A estrutura `std::vector` é um vetor dinâmico que tem funcionalidades parecidas com a lista de Python ou o `ArrayList` de Java. O código abaixo exemplifica seu uso e mostra algumas de suas funções. Note que omitimos o uso de `std` no código abaixo.

```cpp
int n;
cin >> n;
vector<double> vec;
for (int i = 0; i < n; i++) {
    vec.push_back(i * i)
}
cout << "Tamanho do vetor: " << vec.size() << "\n";
cout << "Primeiro elemento: " << vec.front() << "\n";
cout << "Último elemento: " << vec.back() << "\n";
cout << "Elemento 3: " << vec[2] << "\n";
```

Alguns pontos interessantes deste exemplo:

1. Não sabemos o tamanho de `vec` ao criá-lo. O método `push_back` aumenta ele quando necessário e não precisamos nos preocupar com isso. 
2. O número de elementos colocados no vetor é retornado pelo método `size()`
3. O acesso é feito exatamente igual ao array de C, usando os colchetes `[]`

!!! tip "E esse `<double>` na declaração?" 
    Em C++ tipos passados entre `< >` são usados para parametrizar tipos genéricos. Ou seja, um vetor pode guardar qualquer tipo de dado e precisamos indicar qual ao criá-lo. 

    Note que, portanto, um vetor `vector<int>` e um vetor `vector<double>` são considerados de tipos diferentes e não posso passar o primeiro para uma função esperando o segundo. 

!!! question
    Modifique sua Tarefa 4 para usar `vector`. Meça o desempenho com o programa `time` e anote abaixo seus resultados. 