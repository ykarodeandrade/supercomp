# 14 - Mais números aleatórios em GPU

Na última aula trabalhamos com algoritmos simples que faziam sorteios aleatórios. Neste aula exploraremos novamente o 

# Parte 0 - revisão do algoritmo sequencial 

O algoritmo sequencial se baseia em sorteios de pontos dentro de um quadrado de lado `2`. Se a distância entre o ponto e o centro do quadrado for menor que 1 então o ponto cai dentro do círculo inscrito no quadrado. A quantidade de pontos que caem dentro do quadrado é proporcional a $\pi$. 

1. `sum = 0`
1. De `i=0` até `N`:
    1. sorteie pontos $x,y \in [0,1]$
    1. se $x^2 + y^2 \leq 1$, `sum += 1`
1. devolva `4 * sum / N`

# Parte 1 - implementações inocentes

Vamos iniciar nossa implementação fazendo uma implementação ingênua do programa acima. Primeiro iremos sortear todos os pares de pontos. Então, iremos usar uma operação customizada para fazer a comparação acima e fazer a soma final.

!!! example
    Crie um programa que gera `N=100 000` pares de números aleatórios no intervalo `[0,1]`. Use a **Abordagem 1** da aula passada (uma *seed* por thread). Armazene as componentes $x$ em um vetor `rx` e as componentes $y$ em um vetor `ry`.

!!! example
    Faça uma transformação customizada que cria um vetor `dentro` em que cada posição `i` contem `1` se `x[i], yi[i]` satisfazem a condição da parte 0 e 0 caso contrário. Use este vetor para calcular o $\pi$

!!! example 
    Podemos usar acesso direto a elementos do vetor para eliminar o vetor temporário do exercício acima.  Faça isto e compare os tempos de execução. 

    **Dica**: use `transform_reduce`

# Parte 2 - economizando memória e calibrando o *RNG*

Uma grande desvantagem do programa acima é que ele gasta muita memória. Um `transform` é feito só para gerar os números aleatórios e armazenar nos vetores `rx, ry`! Poderíamos fazer isto direto na transformação customizada do último exercício!

!!! example
    Crie uma nova versão do seu exercício anterior, mas agora faça a geração do par $x,y$ e a comparação em uma chamada só. Ou seja, você estará juntando seus dois `struct`s em um só. Seu `struct` agora receberá índices e não mais as componentes $(x, y)$ do ponto. 

Agora nosso programa está mais eficiente: faz somente uma chamada à GPU (`transform_reduce`) e não usa memória auxiliar. Ainda assim há dois problemas: 

1. uma thread é aberta só para sortear dois números e fazer uma comparação. Ou seja, geramos `N=100 000` threads que fazem pouquíssimo trabalho. Isto pode significar que o custo de gerar todas essas threads seja grande perto do tempo que passamos realmente calculando o que precisamos!
2. o *RNG* está gerando somente um número por *seed*.

Apesar do primeiro item não ser exatamente um grande o problema (pode ser que fique mais rápido se balancear), o segundo item afeta diretamente a precisão do nosso programa!

Vamos realizar um experimento rápido e testar se a quantidade de números sorteados pelo *RNG* afeta os resultados. 

!!! example
    Modifique seu `struct` para receber um número `N_thread` que representará o número de sorteios feitos por cada thread. Teste seu programa com `N_thread=1` e `N=100000` e verifique que ele continua funcionando. 

Vamos agora executar o experimento

!!! question 
    Com `N= 100 000`, escreva abaixo o valor do `pi` para cada valor de `N_thread`:

    * `1`: ____
    * `10`:____
    * `100`:_____

    Não se esqueça de calibrar também o número `N_parts` de threads criadas! Ou seja, `N == N_parts * N_threads`.

Esses resultados ocorrem pois o *RNG* garante que sua sequência parece aleatória, não que o primeiro número de cada sequência será aleatório! Ou seja, **só conseguimos as propriedades de aleatoriedade se, para cada *seed*, gerarmos uma quantidade grande de números!**

!!! question 
    Com `N= 100 000`, escreva abaixo o tempo total do programa para cada valor de `N_thread`:

    * `1`: ____
    * `10`:____
    * `100`:_____

!!! warning 
    A primeira execução de um programa em GPU é muito lenta pois ela inclui um tempo de compilação e transferência do programa para a GPU. Os tempos para as execuções subsequentes devem ser bem menores e ter menos variância.

Por outro lado, GPUs são muito boas em criar grande quantidade de threads e executá-las. Ou seja, quanto mais trabalho é feito por cada thread menos vantajoso será sua paralelização em GPU! Como visto acima, o tempo aumenta proporcional a quantidade de trabalho feito nas threads. 


