# 13 - Números aleatórios em GPU

A última etapa da nossa formação em *GPU* é a utilização de geradores de números pseudo-aleatórios. Isso é uma etapa importante para implementação de algoritmos de simulação e otimização. 

## Parte 0 - revisão de números aleatórios

Um gerador de números pseudo-aleatórios (RNG) é um algoritmo **determinístico** que gera uma sequência de números que **parece** aleatória. Essa frase possui dois termos importantes que precisamos destrinchar:

* **determinístico**: Um *RNG* tipicamente recebe como entrada um inteiro *seed* (que representa uma sequência de bits "aleatória") e gera uma sequência de números baseada no *seed*. Ou seja, o algoritmo é **determinístico** pois gera sempre a mesma sequência para uma determinada entrada (*seed*).
* **parece aleatória**: Se compararmos duas sequências de números, uma gerada por um *RNG* e outra por uma distribuição uniforme de verdade, não conseguimos dizer qual distribuição foi gerada pelo *RNG*. 

Ou seja, ao escolhermos um *seed* a sequência gerada será sempre a mesma, mesmo se executarmos o programa em outras máquinas. Isso torna a utilização de *RNGs* para experimentos bastante interessante: é possível **reproduzir** os resultados feitos por outros desenvolvedores/cientistas. Para isto é necessário

1. que o programa permita escolher o *seed* da simulação;
1. que o *seed* usado seja publicado junto com os resultados.

!!! question short
    E se quisermos gerar uma sequência diferente a cada execução do programa? Como poderíamos configurar o *seed* para que isto aconteça?

Muitas implementações de *RNGs*  são divididas em duas partes:

1. **engine**: algoritmo que gera um inteiro cujos bits formam uma sequência pseudo-aleatória.
1. **distribution**: utiliza os bits acima para retornar números que sigam alguma distribuição estatística (como normal ou uniforme).

As classes do cabeçalho `<random>` seguem este padrão:

1. `std::random::default_random_engine` gera bits aleatórios
1. todas as funções `std::random::*_distribution`. Cada um dos exemplos abaixo gera números seguindo suas respectivas distribuições estatísticas e recebem os parâmetros de cada distribuição. 
    * `poisson_distribution`
    * `uniform_int_distribution`
    * `uniform_real_distribution`
    * `normal_distribution`.

Vamos agora verificar na prática essas propriedades criando um programa de testes. 

!!! example
    Crie um programa que leia um inteiro *seed* do terminal e:

    1. crie um objeto `default_random_engine` que o utilize como seed.
    1. mostre no terminal uma sequência de 10 números fracionários tirados de uma distribuição uniforme `[25, 40]`.

## Parte 1 - `thrust` e *RNG*

A `thrust` contém um

!!! question short
    Consulte a documentação oficial da `thrust` e encontre as páginas que descrevem os **engines** e **distributions** implementados. 

Vamos agora repetir o exercício da parte anterior em `thrust`. 

!!! example
    Crie um programa que leia um inteiro *seed* do terminal e:

    1. crie um objeto `default_random_engine` que o utilize como seed.
    1. mostre no terminal uma sequência de 10 números fracionários tirados de uma distribuição uniforme `[25, 40]`.

    Seu programa deverá estar implementado usando os tipos definidos em `thrust::random`.

Um ponto importante da API `thrust` para geração de números aleatórios é que essas funções podem ser chamadas dentro de operações customizadas! Vamos continuar trabalhando com imagens no próximo exercício, mas desta vez faremos uma operação exatamente contrária: adicionaremos ruído a uma imagem. 

Usaremos o seguinte algoritmo: para cada ponto da imagem sortearemos um número entre `1` e `10` inclusive. 

1. Se o número for `1` a cor atual deve ser substituída por preto. 
1. Se o número for `10` a cor atual deve ser substituída por branco.
1. Caso contrário não mexa na cor atual. 

!!! example
    Crie um programa que recebe uma imagem como argumento e escreva em uma segunda imagem o resultado do algoritmo de ruído acima. Seu programa deverá funcionar como abaixo:

    `$> ruido in.pgm out.pgm`

!!! warning
    Mesmo que seu programa aparentemente não funcione, valide sua saída com o professor.

## Parte 2 - seeds em programas aleatórios

Um desafio em programas paralelos é gerar sequências pseudo-aleatórias de qualidade. Se não tormarmos cuidado acabamos gerando os mesmos números em threads diferentes e desperdiçamos grande quantidade de trabalho! Em geral existem duas abordagens 

**Abordagem 1**: usar *seeds* diferentes em cada thread. 

**Abordagem 2**: usar a mesma *seed* em todas as threads, mas cada uma começa em um ponto diferente da sequência daquela seed.

Note que em ambos os casos os resultados dependem do número de threads usadas! Como vimos em aulas anteriores, um *RNG* tem estado interno e não pode ser facilmente compartilhado entre várias threads. 

!!! example
    Implemente a abordagem 1 no exercício da parte anterior. Para isto você pode usar o índice recebido como *seed*

!!! example
    Implemente a abordagem 2 no exercício da parte anterior. Para isto você pode *descartar* os *i* primeiros números aleatórios gerados. Procure na documentação oficial como fazer isto. 

!!! example
    Compare os dois resultados em termos de tempo e quantidade de ruído presente na imagem de saída. 