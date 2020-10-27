# 13 - Paralelismo de dados

Nesta prática iremos usar a contrução `omp parallel for` para tratar casos de paralelismo de dados. 

## O `for` paralelo

Vamos começar nosso estudo do `for` paralelo executando alguns programas e entendendo como essa construção divide as iterações entre threads. 

!!! question short
    Você consegue predizer o resultado do código abaixo? Se sim, qual seria sua saída? Se não, explique por que. 

    ```c
    #pragma omp parallel for
    for (int i = 0; i < 16; i++) {
        std::cout << "Eu rodei na thread: " << omp_get_thread_num() << "\n";
    }
    ```

!!! example
    O código acima está no programa *exemplo1.cpp*. Execute-o várias vezes e veja se sua resposta acima é condizente com a realidade.

    ??? details "Resposta"
        Não é possível predizer. No caso acima o loop foi dividido igualmente entre as threads, mas isso é uma decisão do compilador e não temos controle sobre qual será seu comportamento. Isso pode variar de compilador para compilador.

        O comportamento automático funciona bem na maioria das vezes. 

!!! question medium
    Examine o código abaixo e responda.

    ```c
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < 16; i++) {
        std::cout << "Eu rodei na thread: " << omp_get_thread_num() << "\n";
    }
    ```

    1. Quantos cores, no máximo, serão usados?
    2. Você consegue dizer em qual thread cada iteração rodará?
    3. Você consegue dizer quantas iterações cada thread rodará?
    4. Suponha que a thread 4 iniciou a iteração `i=4`. Ela processará somente essa iteração isoladamente? Se sim, explique por que. Se não, diga até qual valor de `i` ela executará.
    5. As alocações mudam a cada execução do programa?


!!! example
    O código acima está no programa *exemplo2.cpp*. Execute-o várias vezes e veja se sua resposta acima é condizente com a realidade.

!!! question medium
    Examine o código abaixo e responda.

    ```c
    #pragma omp parallel for schedule(static, 4)
    for (int i = 0; i < 16; i++) {
        std::cout << "Eu rodei na thread: " << omp_get_thread_num() << "\n";
    }
    ```

    1. Quantos cores, no máximo, serão usados?
    2. Você consegue dizer em qual thread cada iteração rodará?
    3. Você consegue dizer quantas iterações cada thread rodará?
    4. As alocações mudam a cada execução do programa?

!!! example
    O código acima está no programa *exemplo3.cpp*. Execute-o várias vezes e veja se sua resposta acima é condizente com a realidade.

Vamos agora aplicar esses conhecimentos no exemplo do cálculo do *pi* da aula passada. 

!!! example 
	Modifique o código sequencial para usar as construções `parallel for` e `reduce`. Os resultados se mantiveram iguais? E o tempo?

!!! example
    Multiplique `num_steps` por 10 e tente novamente. E agora? Os ganhos são mais aparentes?

Se você achou fácil é por que é mesmo. *OpenMP* é ideal para situações como esta:

1. pouca ou nenhuma dependência de dados
2. pouca ou nenhuma sincronização
3. loop que roda uma grande quantidade de vezes o mesmo código. 

Também vimos a maior vantagem do *OpenMP*: uma pequena modificação no código produz paralelismo eficiente. O maior desafio é **entender onde estão as oportunidades de paralelismo**. Com isso identificado, adicionar as anotações no código é rápido. 

## Exercício prático

Vamos agora trabalhar com imagens e paralelizar alguns processamentos de imagem. A ideia será programar alguns processamentos simples e verificar se o OpenMP traz ganhos de desempenho. Nossa primeira operação será **adicionar ruído a uma imagem**. Isto é um pré-processamento comum feito para testar a robustez de algoritmos de processamento de imagens a pequenos defeitos.

Nosso algoritmo será o seguinte: para cada ponto da imagem sortearemos um número entre `1` e `10` inclusive.

1. Se o número for `1` a cor atual deve ser substituída por preto.
1. Se o número for `10` a cor atual deve ser substituída por branco.
1. Caso contrário não mexa na cor atual.

!!! question
    Examine o cabeçalho *imagem.hpp* e verifique como usar a classe *Imagem*. Onde estão armazenados o tamanho da imagem? E os pontos? Como acessar o elemento *i, j* da imagem?
    Qual é o valor de um pixel branco? E de um pixel preto?

!!! example
    Faça uma versão sequencial do algoritmo acima, colocando sua solução no arquivo *exercicio1.cpp*. Teste-a com as imagens da pasta atual e veja se as saídas incluem pontos brancos/pretos.

Vamos agora tentar paralelizar seu programa acima.

!!! question short
    Identique quais partes do programa são paralelizáveis. Existe alguma dependência de dados? Qual?

!!! question short
    É possível desfazer a dependência acima? Ou ao menos evitar que ela estrague nosso programa? Como você faria isso?

    ??? details "Resposta"
        O gerador de números aleatórios depende da ordem de uso para funcionar! Se as iterações rodarem fora de ordem então o resultado do nosso programa será imprevisível!

!!! example
    Antes de prosseguir, tente paralelizar seu programa usando OpenMP. Por enquanto, ignore as dependências identificadas acima e finja que tudo dará certo. 

!!! question short
    Execute o programa paralelo duas vezes. Os resultados são idênticos? E se executar o programa sequencial?

Nossa solução será baseada em "enganar" o gerador de números aleatórios para que ele continue gerando números a partir de um certo local de sua sequência.

!!! question short
    Na iteração `i=300` do seu código anterior, quantos números aleatórios já foram gerados?

    ??? details "Resposta"
        Já foram gerados 299 números. 

!!! question short
    Veja a documentação do método `discard` de [default random engine](http://cplusplus.com/reference/random/default_random_engine/). Como esse método pode nos ajudar?

    ??? details "Resposta"
        Podemos criar um novo gerador a cada passo e avançá-lo para onde ele estaria no código sequencial.

!!! example
    Implemente o programa da resposta acima. Você deverá usar `discard` para que os resultados sejam idênticos aos do programa sequencial.

!!! danger
    Se seu programa ficou muito mais lento mas resultados idênticos, prossiga.

!!! question short
    Volte na documentação de `discard` e procure por sua complexidade computacional. Você consegue explicar a razão do programa ter ficado mais lento?

