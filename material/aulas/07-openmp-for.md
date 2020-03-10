# 07 - OpenMP: Construções de alto nível

Agora que vimos as construções de `for` paralelo e de redução iremos aplicá-las no nosso exercício do cálculo do pi. Por trás dos panos essas construções fazem o mesmo trabalho dos códigos que criamos anteriormente, porém não precisamos escrever tanto código. 

!!! example 
	Modifique sua solução para usar as construções `parallel for` e `reduce`. Coloque seu resultado na função `double pi_omp_for(long steps)`. Seus resultados se mantiveram iguais? 

Se seu exercício deu certo você deve ter obtido tempos comparáveis com as melhores implementações anteriores, porém seu programa atual é praticamente igual ao sequencial. Esta é a principal vantagem de trabalhar com OpenMP: com poucas modificações conseguimos transformar um código sequencial em um programa paralelo. 

# Paralelizando um programa já existente. 

A atividade hoje será debugar um programa paralelo que deu errado! 

**Curiosidade**: fractais são estruturas matemáticas que são definidas por sua auto-similaridade. Eles são úteis para modelar objetos e fenômenos que possuem as mesmas características em escalas completamente diferentes, como nuvens, montanhas e [compressão de arquivos](https://en.wikipedia.org/wiki/Fractal_compression). 

![Exemplos de fractais, Fonte: mathworld.wolfram.com.](http://mathworld.wolfram.com/images/eps-gif/Fractal1_1000.gif)

----

O arquivo *mandel.cpp* tem uma implementação que calcula a área abaixo do [Fractal de Mandelbrot](https://en.wikipedia.org/wiki/Mandelbrot_set). Foi feita uma tentativa preguiçosa de paralelização usando OpenMP que está dando resultados muito estranhos. 

!!! example 
	Este código tem problemas de compartilhamento indevido de dados. Você deverá 
	
	1. modificar o `pragma` que paraleliza o `for` para que não haja compartilhamento indevido de variáveis. 
	2. verificar se existe compartilhamento não intencional das variáveis globais e arrumar, caso necessário.  

	Você deve modificar o programa o mínimo possível 


### Teoria - Efeitos colaterais 

Dizemos que quando uma função escreve/lê em variáveis globais (ou `static`) ela possui *efeitos colaterais*. Ou seja, após rodar ela modifica o estado do programa. Em comparação, uma função que só depende dos valores passados nos argumentos e não escreve seu resultado (ou valores intermediários) em variáveis globais é dita *sem efeitos colaterais*. Este tipo de função pode ser chamado por várias threads simultâneamente e é uma boa prática de programação paralela criar funções sem efeitos colaterais. 

!!! example
	Apesar do programa agora funcionar, os erros do exercício acima eram causados essencialmente por más práticas de programação. Reestruture o código para que o resultado de suas funções só dependa dos valores passados nos argumentos. Isto costuma implicar na conversão de valores lidos/escritos em variáveis globais para valores passados nos parâmetros da função. Neste exercício você pode mudar o programa extensamente, desde que os resultados continuem os mesmos.  



