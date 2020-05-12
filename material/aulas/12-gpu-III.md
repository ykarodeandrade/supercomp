# 12 - introdução a GPU - III

Hoje o foco será operações customizadas. 

## Parte 0 - revisão

Na aula anterior fizemos várias implementações da variância. Podemos fazer uma versão ainda mais sucinta usando operações customizadas. Um truque comum é adicionar atributos no `struct` usado como operação:

```cpp
struct T {
    int attr;

    T(int a): attr(a) {};

    // TODO: operação customizada aqui
};
```

!!! example
    Faça uma nova implementação da variância, dessa vez usando uma operação customizada e com a chamada `transform_reduce`. 

    **Dica**: passe a média e o tamanho do vetor como atributo do `struct`.

## Parte 1 - acesso direto a vetores

Apesar da `thrust` nos permitir acessar os dados de cada iteração, o acesso a elementos arbitrários do vetor não é diretamente suportado. Apesar de ser possível fazer isto com iteradores e tuplas, vamos usar uma abordagem um pouco diferente: acessar o vetor diretamente e usar a `thrust` para fornecer ao nossa transformação customizada o índice a ser usado. Vejamos abaixo um exemplo simples:


```cpp
--8<--- "12-gpu-III/raw_access.cu"
```

O `struct raw_access` recebe um ponteiro para o vetor e guarda em um atributo. Os valores recebidos em `operator()` são simplesmente os índices do vetor a serem tratados. Com isso, podemos fazer modificações complexas em vetores trabalhando apenas com índices e acessos diretos a memória.

!!! warning 
    A `thrust` é bastante eficiente e os algoritmos foram implementados por especialistas em GPGPU. Prefira usar os recursos disponíveis na biblioteca se possível.

!!! example
    Reimplemente o cálculo das diferenças adjancente usando uma transformação customizada. Aproveite e já zere toda diferença negativa no novo vetor. 

## Parte 2 - acesso a matrizes

Assim como na CPU, podemos representar imagens como um vetor "deitado". O acesso ao elemento `(i, j)` é feito como abaixo.

```cpp
img[i * width + j] = 10;
```

!!! question short
    Examine os arquivo *imagem.cpp/h*. Quais funções são definidas e o quê elas fazem?

O filtro de média é um processamento de imagens simples muito usado para tirar suavizar imagens. Sua implementação é bastante simples. Dada uma imagem de entrada $I$, a imagem de saída $O$ é dada pela seguinte expressão.

$$
O[i, j] = \frac{I[i, j] + I[i-1, j] + I[i+1, j] + I[i, j-1] + I[i,j+1]}{5}
$$

Ambas as imagens tem o mesmo tamanho. Se o pixel acessado estiver fora da área válida da imagem ele deve ser considerado 0. 

!!! note
    Você pode converter qualquer imagem que tiver em seu computador para *PGM* usando o programa `convert`:

    `$> convert imagem.png -compress None imagem.pgm`

!!! example
    Implemente um programa `media_gpu` que faz o processamento descrito acima usando `thrust`. Seu programa deverá funcionar como abaixo. 

    `$> media_gpu in.pgm out.pgm`

!!! example
    Faça uma implementação em CPU do filtro de média. Chame-a de `media_cpu`. Ela deverá funcionar de maneira idêntica ao programa acima.

!!! example
    Teste seu programa com diferentes tamanhos de imagens e compare os tempos de execução. A partir de qual tamanho os tempos ficam equivalentes?
