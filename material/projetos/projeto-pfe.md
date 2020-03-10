# Projeto semestral - Alocação de alunos para o PFE

Neste semeste trabalharemos com um problema bastante familiar para os alunos do 8/9 semestre: a atribuição de alunos a projetos no Projeto Final de Engenharia.

* Cada aluno escolhe cinco opções de projetos de acordo com sua preferência.
* Cada projeto receberá exatamente três alunos.

Uma solução para este problema é uma atribuição de três alunos para cada projeto. Claramente algumas soluções são melhores pois maior quantidade de alunos está alocada em projetos que tem maior preferência. Para quantificar esta qualidade de cada solução vamos adotar a seguinte estratégia:

1. $5^2 = 25$ se foi colocado na primeira opção
1. ....
1. $2^2 = 4$ se foi colocado em sua quarta opção
1. $1$ se foi colocado em sua última opção
1. $0$ se não foi possível colocá-lo em nenhuma de suas opções. 
* a satisfação "global" de uma solução é a soma da satisfação individual de todos alunos. 

Desta maneira, gostaríamos de encontrar a melhor solução (a que possui maior satisfação global) dados o número de alunos, o número de projetos e até 5 opções de projetos em ordem de preferência para cada aluno.

## Justificativa

Apesar de parecer simples, este problema é *NP-completo*: não existe um método que encontra a solução com maior satisfação global em tempo polinomial. Além disso, não existe uma maneira que verifique se uma solução é a melhor possível em tempo polinomial. Ou seja, trabalhar com este problema invariavelmente envolverá enumerar todas as possibilidades. Claramente este é um problema onde SuperComputação é necessária! Veremos que técnicas de computação paralela podem diminuir consideravelmente o tempo de execução de nosso programa. 

Outra fonte de ideias para acelerar a resolução deste problema é a utilização de técnicas de otimização discreta, que são nada mais do que explorar alguma característica do problema que estamos tratando para melhorar nossas soluções. Podemos usá-las para 

* encontrar boas soluções sem enumerar todas **ou** 
* para evitar enumerar soluções que com certeza não são as ótimas. 

Porém, nem sempre é fácil paralelizar estas técnicas e este será um dos desafios dos projetos deste semestre.

## Entrada e saída

O formato de entrada do programa estará no formato abaixo.

```
n_alunos n_projetos
p1 p2 p3 p4 p5
..... # repetido n_alunos vezes
```

* `n_alunos` é o número de alunos
* `n_projetos` é o número de projetos existentes
* cada linha seguinte representa as cinco prioridades de um aluno.    
* `p1, ..., p5` é um número entre `0` e `n_projetos-1`, sem repetições

A saída do programa deverá estar no formato abaixo.

```
satisfacao opt
pa1 pa2 pa3 ... pa(n_alunos)
```

* `satisfacao`
* `opt` é `1` se a solução encontrada é a melhor possível, `0` caso contrário
* `pa(i) ` contém a qual projeto o aluno `i` foi atribuído

A pasta [code/projeto](...) contém exemplos de entradas e saída esperadas. Seu programa deverá funcionar com estas entradas e produzir saídas **exatamente** neste formato. 

!!! note "Avisos"
    * Se existirem duas soluções com mesma satisfação qualquer uma pode ser retornada.  
    * Informações de *debug* devem ser enviadas para a saída de erros (`std::cerr`). A saída de seu programa deverá estar **exatamente** no formato mostrado na seção anterior.

### Simplificações:

Seu programa pode assumir o seguinte:

1. `n_alunos` é divisível por 3
1. `n_projetos` é exatamente `n_alunos/3`

Isto facilita muito o problema, pois nunca conseguimos montar uma solução inválida nem precisamos levar em conta a distribuição de alunos ao criar soluções.

## Entregas

Os projetos da disciplina envolverão resolver este problema usando diferentes tecnologias e iremos comparar os desempenhos obtidos com cada uma. 





