# 02 - Detalhes de implementação

Nesta aula trabalharemos dois objetivos:

1. implementação de algoritmos dada uma descrição de alto nível da tarefa a ser implementada
2. técnicas de implementação para alto desempenho


!!! warning "Software"
    Para esta aula precisaremos dos seguintes pacotes instalados. 

    * `valgrind` - ferramenta de análise de código executável
    * `kcachegrind` - visualizador de resultados do `valgrind`

## O problema básico

Dados `N` pontos com coordenadas $(x_i, y_i)_{i=0}^N$, computar a matriz de distâncias $D$ tal que 

$$
D_{i,j} = \textrm{Distância entre } (x_i, y_i) \textrm{ e } (x_j, y_j)
$$

!!! example
    Implemente um programa que calcule a matriz `D` acima. Sua entrada deverá estar no formato dos arquivos `t1-in-*.txt` e sua saída no formato dos arquivos `t1-out-*.txt`. 

    **Dicas**:
    
    1. a maneira mais fácil (não necessariamente a melhor) de alocar uma matriz é usando um vetor em que cada elemento é outro vetor. 
    2. faça uma implementação o mais simples possível. Vamos melhorá-la nas próximas tarefas.

    ??? details "Resposta"
        ```
        leia inteiro N
        leia vetores X e Y 

        seja D uma matriz NxN

        para i=1..N:
            para j=1..N:
                DX = X[i] - X[j]
                DY = Y[i] - Y[j]
                D[i,j] = sqrt(DX*DX + DY*DY)
        ```


!!! question medium
    Anote abaixo o tempo de execução para os arquivos `t1-in-*.txt` e `t1-out-*.txt`

!!! question 
    Qual é a complexidade computacional de sua implementação? 

## Passagem de dados no programa

Na parte anterior fizemos nosso programa inteiro no `main`. Vamos agora organizá-lo melhor. 

!!! example
    Crie uma função `calcula_distancias` que recebe a matriz e os dados recebidos na entrada e a preenche. Sua função não deverá retornar nenhum valor. 

    Ao terminar, meça o tempo de execução para o arquivo `t1-out-4.txt`.

    ??? details "Resposta"
        Aqui podem ocorrer dois problemas:

        1. Seu programa deu "Segmentation Fault". 
        2. Seu programa rodou até o fim, mas a saída é vazia (ou cheia de 0).

        O problema em si depende de como você fez o `for` duplo para mostrar os resultados. De qualquer maneira, simplesmente mover código para uma outra função não funciona neste caso. 

Ambos problemas descritos na solução são previsíveis e ocorrem pela mesma razão: **ao passar um `vector` para uma função é feita uma cópia de seu conteúdo**. Ou seja, a matriz usada dentro de `calcula_distancias` não é a mesma do `main`! 

Isto é considerado uma *feature* em `C++`: por padrão toda variável é passada **por cópia**. Isto evita que uma função modifique um valor sem que o código chamador fique sabendo. 

Em *C* podemos passar variáveis **por referência** passando um ponteiro para elas. Apesar de funcional, isso não é muito prático pois temos que acessar a variável sempre usando `*`.  Em *C++* temos um novo recurso: referências. Ao declarar uma variável como uma referência crio uma espécie de *ponteiro constante* que sempre acessa a variável apontada. Veja o exemplo abaixo.

```cpp
int x = 10;
int &ref = x; // referências são declaradas colocando & na frente do nome da variável
// a partir daqui ref e x representam a mesma variável
ref = 15;
cout << x << "\n"; // 15
```

O mesmo poderia ser feito com ponteiros (como mostrado abaixo). A grande vantagem da referência é que não precisamos usar `*ref` para nos referirmos à variável `x`! Na atribuição também podemos usar direto `int &ref = x`, o que torna o código mais limpo e fácil de entender.  

```cpp
int x = 10;
int *ref = &x; // precisamos de &x para apontar ref para a variável x
*ref = 15; // precisamos indicar *ref para atribuir a variável x
cout << x << "\n"; // 15
```

!!! tip "Dicas"
    Note que uma referência tem que ser inicializada com a variável a que ela se refere. Ou seja, ao declarar tenho que já indicar a variável destino. 

!!! example
    Modifique sua função para usar referências. Verifique que ele volta a funcionar e que seu tempo de execução continua parecido com a versão que rodava no `main`.

    ??? details "Resposta"
        Basta adicionar `&` na frente dos nomes dos argumentos (vetores x, y e matriz). A chamada da função não muda. 

!!! tip "Dica"
    Em *C++* precisamos estar sempre atentos a maneira que passamos os dados. Se não indicarmos será por cópia. Para compartilhar o mesmo objeto entre várias funções usamos referências `&`. 

## Medição de tempo com KCachegrind

Apesar de podermos medir o tempo que nosso programa demora usando o comando `time`, não conseguimos nenhuma informação importante de qual parte do programa está consumindo mais tempo. Este processo de dissecar um programa e entender exatamente qual parte demora quanto é chamada de **Profiling**. 

!!! tip "Dica"
    É preciso compilar um executável com profiling habilitado para medir os tempos. Isso é feito com a flag `-p` do `g++`. Veja abaixo. 

    ```
    g++ -p -g euclides-ingenuo.cpp -o euclides-ingenuo
    ```

??? quote "Demonstração"
    adslj
    lkadj

    ```

    ```

    Para mostrar os resultados usando o `kcachegrind` usamos o seguinte comando. 

    ```
    kcachegrind callgrind.out.(pid aqui)
    ```

Na demonstração pudemos ver que grande parte do tempo do programa da Tarefa 1 é gasto mostrando a saída no terminal. Isto nos leva à primeira conclusão da atividade de hoje:

!!! info "Entrada e saída de dados são operações muito lentas"

Com isso em mente, vamos agora otimizar a função `calcula_distancias`. Já sabemos que o efeito no tempo final não será grande. Nosso objetivo então será verificar a seguinte afirmação. 

!!! info "Dois algoritmos de mesma complexidade computacional podem ter tempos de execução muito diferentes"

A otimização que trabalharemos nesse roteiro tentará explorar a **simetria** da matriz `D`. 

!!! question
    Como isso poderia ser usado para melhorar o tempo de execução de `calcula_distancias`?

!!! question 
    Seu programa criado na tarefa 1 consegue ser adaptado para implementar sua ideia da questão 
    anterior? O que precisaria ser modificado?

    ??? note "Resposta"
        Duas respostas são possíveis e corretas aqui:

        1. Preciso checar se o `i < j` e usar o valor já calculado de `D[j,i]`.
        
        2. É preciso alocar a matriz inteira antes de começar. Se formos dando `push_back` linha a linha não conseguimos atribuir um valor ao mesmo tempo a `D[i,j]` e `D[j,i]`, já que um deles ainda não terá sido criado. 

Vamos começar implementando resposta 1 da pergunta anterior, já que ela envolve uma pequena modificação no programa. 

!!! question
    Anote abaixo o consumo absoluto de tempo da função `calcula_distancia`.

!!! example
    Adicione uma checagem que verifica se o elemento já foi calculado no "outro lado" da matriz e use esse valor ao invés de 

!!! question
    Anote abaixo o consumo absoluto de tempo da sua nova função `calcula_distancia`. Compare com a pergunta anterior e tente entender seus resultados.

    ??? note "Resposta"
        Deve ter havido uma pequena melhora, mas longe de ser a metade do tempo. 

!!! question
    Por que não houve melhora significativa? Você consegue explicar?

    ??? nota "Resposta"
        A principal razão é que o número absoluto de instruções não mudou muito. O `for` duplo ainda roda `n*n` vezes e o próprio acesso a `mat[i][j]` é apontado como lento pelo `kcachegrind`. 

Com a alternativa 1 descartada, vamos agora para a alternativa 2: atribuir de uma só vez em `D[i,j]` e `D[j,i]`. Ou seja, agora nosso loop vai ser executado metade das vezes! 

!!! question
    Supondo que a matriz já esteja inteira alocada, como você implementaria a alternativa 2?

    ??? note "Resposta"
        ```
        para i=1..N:
            para j=i..N:
                DX = X[i] - X[j]
                DY = Y[i] - Y[j]
                DIST = sqrt(DX*DX + DY*DY)
                D[i,j] = DIST
                D[j,i] = DIST
        ```

