# 01 - Aquecimento

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