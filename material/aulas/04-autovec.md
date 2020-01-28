
# 04 - SIMD: Autovetorização


Programas de fazer cálculos podem ser acelerados significativamente por instruções *SIMD* (*Single Instruction Multiple Data*). Neste contexto, a utilização de compiladores com opção de *autovetorização* é muito conveniente: se o código compilado tiver algumas características o compilador é capaz utilizar estas instruções automaticamente e melhorar muito o desempenho sem que um programador precise modificar código. Um guia para autovetorização usando `gcc` pode ser encontrado em [sua documentação](https://gcc.gnu.org/projects/tree-ssa/vectorization.html).

!!! info
    Os exercícios de aula podem ser feitos em dupla.

## Capacidade dos registradores

!!! question
    Levando em conta que a arquitetura *AVX* armazena dados em registradores de 128 *bits*, podemos armazenar em um único registrador até:

    * ____ chars;
    * ____ shorts;
    * ____ ints;
    * ____ longs;
    * ____ floats;
    * ____ doubles;

!!! question
    A arquitetura *AVX2* suporta registradores de 256 *bits*. Isto significa que código autovetorizado com instruções *AVX2* pode ser até ____ vezes mais rápido do que código vetorizado com *AVX*
    
!!! question
    Toda instrução SIMD opera sobre todos os elementos guardados ao mesmo tempo. Ou seja, ao executar uma instrução SIMD de soma de variáveis `int` no registrador `%xmm0` estarei somando ____ variáveis em uma só instrução.


    
## Instruções vetoriais

Vamos agora compilar alguns códigos em C++ para Assembly usando 
as opções `-ffast-math -ftree-vectorize -mavx` do `gcc`. Estas opções habilitam a *autovetorização*
de código. Ou sejam elas analisam o código e procuram substituir loops e acessos sequenciais
a vetores por instruções SIMD que façam o mesmo trabalho.

A flag `-mavx` indica que o código de máquina gerado pode utilizar instruções
SIMD da arquitetura *AVX* (e de sua predecessora, SSE), que usa registradores de 128 bits nomeados `%xmm0` 
até `%xmm7` e de 256 bits nomeados `%ymm0` até `%ymm15`. 
    
    
Vamos agora analisar o código Assembly de uma função simples que soma todos elementos de um vetor. 

```cpp
--8<-- "04-autovec/tarefa1.cpp"
```

!!! example
    Compile este código com otimizações básicas (`-O2`)


    >$ g++ -S -c -O2  tarefa1.cpp -o /dev/stdout | c++filt

    Agora adicione autovetorização (com as flags listadas acima)
    
    >$ g++ -S -c -O2 -ffast-math -ftree-vectorize -mavx tarefa1.cpp -o /dev/stdout | c++filt

!!! question long
    Compare as instruções Assembly geradas acima e escreva abaixo as diferenças percebidas. Você consegue explicar seu funcionamento?

## Aplicando autovetorização

Nesta tarefa iremos trabalhar com as opções de autovetorização do `gcc` para entender como escrever código que possa ser otimizado automaticamente. 

!!! example
    Escreva uma função `main` que gera um vetor de tamanho `10.000.000` contendo números aleatórios uniformemente distribuídos entre `-10` e `10`. Use as funções do cabeçalho `<random>`. 

!!! example 
    Escreva uma função `double soma_positivos1(double *a, int n)` que soma todos os números positivos do vetor `a`. Adicione uma chamada a esta função no seu `main` e use as funções do cabeçalho `<chrono>` para medir o tempo de execução da sua função. Neste exercício você deverá usar um `if` para checar se os números são positivos. 

!!! question short
    Compile com e sem as otimizações SIMD e escreva abaixo os tempos de execução.

O auto vetorizador suporta uma série de padrões de codificação relativamente abrangente ([lista completa](https://gcc.gnu.org/projects/tree-ssa/vectorization.html)). Porém, códigos que são vetorizados de maneira idêntica podem possuir desempenho bastante diferente quanto a vetorização não está habilitada.

!!! example
    Faça uma função `double soma_positivos2(double *a, int n)` que faz o mesmo que a função anterior, mas usando agora o operador ternário `(cond)?expr_true:expr_false` ao invés de um `if`. (Se você fez com o operador ternário acima faça com `if`). Houve diferença de desempenho na versão SIMD? E na versão sem SIMD?

!!! question
    Complete a tabela abaixo com os tempos obtidos
    
       -      | SIMD   | sem SIMD
    --------- | ------ | --------
    if-else   |        | 
    ternário  |        |
    
!!! note medium "Exercício" 
    Qual versão da função anterior você usaria se seu código fosse executado em processadores de baixo custo (Intel Celeron) ou muito antigos (mais de 5 anos)? E se o plano for executar em processadores novos? 


## Projeto 0 - aplicando autovetorização

!!! warning 
    Os exercícios do Projeto 0 são individuais. 

Nas últimas duas tarefas vimos como usar as opções do compilador para gerar instruções SIMD, 
tornando nossos programas mais eficientes. Agora vamos aplicá-las ao nosso exemplo dos experimentos. Você deverá

- [ ] criar um novo target `vector_ops_simd` com as opções de compilação SIMD e a flag de otimização `O3`
- [ ] atualizar seus gráficos de desempenho com o novo experimento
- [ ] comentar os novos resultados, respondendo às seguintes perguntas:
    * A partir de qual tamanho de array o código vetorizado mostra ganhos de desempenho expressivos?
    * Qual é o ganho de desempenho esperado? Leve em conta a arquitetura usada e o tipo de dados usado.
    * Os ganhos de desempenho são consistentes com o esperado?
