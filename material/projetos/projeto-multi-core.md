# Multi-core: C++ e OpenMP

O primeiro projeto consiste em implementar uma solução multi-core para o problema da [Alocação de alunos do PFE](projeto-pfe.md). Para facilitar sua ele será dividido em duas partes.

A primeira parte focará em criar uma solução que resolve este problema este C++ e compará-la com uma solução inocente em Python. Isto inclui estudar métodos de resolução do problema e implementá-los em C++.

A segunda parte tratará da paralelização destes métodos usando as técnicas mostradas em sala de aula. O programa deverá ser escrito usando OpenMP e deverá escalar o mais naturalmente possível conforme o número de cores aumenta.

### Parte 1: estratégias de resolução

O arquivo `solucao-ingenua.py` contém uma solução simplista escrita em Python. Este programa sempre encontra a melhor solução, mas é **extremamente ingênuo**. Por isso, ele também é **extremamente lento**.

**Implementação em C++**:

Nosso primeiro passo será implementar esse mesmo programa em C++. O objetivo desta parte será implementar soluções para o problema sem nós preocuparmos com paralelismo e o primeiro passo para isso é implementar o algoritmo que está em Python em C++.

**Busca local**:

A estratégia de busca local visa encontrar boas soluções em um processo de melhora iterativa. A partir de uma solução inicial (que pode ser aleatória), tentamos aplicar uma heurística (truque) que pode melhorar a solução (mas nunca piorar). Note que isto somente garante que a solução irá melhorar iterativamente, porém não garante que eventualmente chegaremos na melhor solução possível. Além dissto,

1. a solução encontrada muda conforme a solução inicial
1. nem todas as soluções são possíveis de serem encontradas.

Uma boa heurística geralmente é baseada em alguma característica da solução ótima. Para este problema usaremos a seguinte propriedade

> não existe nenhuma dupla de alunos que, se for trocada de projeto, melhora a satisfação global.

Claramente se a solução é a melhor possível então isto não pode acontecer. Nosso algoritmo será

1. escolha uma atribuição aluno-projeto válida aleatoriamente
1. verifique se existe um par de alunos cuja troca de projeto melhore a satisfação global
    * se existir faça a troca e repita o teste acima
    * se não existir retorne a solução atual

Ao repertirmos este algoritmo conseguimos soluções razoáveis muito rapidamente. Ele é uma busca **local** pois seu resultado depende de qual solução inicial foi usada. Nem toda solução inicial resultará no ótimo **global** no fim do processo.

**Branch and Bound**:

Nosso algoritmo simplório no item anterior faz várias escolhas recursivas (*branches*) e atualiza a melhor solução encontrada até o momento. Imagine a seguinte situação:

* em um certo momento temos uma solução  com valor $200$
* ainda faltam 3 alunos para serem alocados.
* a melhor solução já encontrada tem valor $300$.

Note que, mesmo se alocarmos os três alunos para sua primeira opção ficaríamos com uma solução de valor $275 < 300$. Ou seja, não precisamos tentar alocá-los para projetos, pois mesmo que façamos o melhor possível ainda não conseguiremos superar a melhor solução atual!

Um **bound** é uma estimativa otimista para o valor final de uma solução parcial. Ou seja, dado que falta ainda alocar *X* alunos e tenho uma solução de valor *Y*, uma estimativa otimista seria supor que todos serão alocados em sua primeira opção (solução final com valor $< Y + 25X$).

!!! warning
	Um bound é uma **estimativa otimista**. Ou seja, pode não existir uma solução com este valor!

Um algoritmo **branch and bound** leva em conta essas estimativas em seu funcionamento:

* se o **bound** da solução atual for pior que a solução ótima atual, retorna
* continue a recursão caso contrário

Esta técnica evita que nossa recursão entre em *branches* que não tem chance nenhuma de descobrir a melhor solução (pois eles já são piores que uma solução válida conhecida).

**Heurísticas de busca**:

O algoritmo recursivo implementado em Python testa todas as possibilidades de maneira bastante inocente. Ele não leva em conta, por exemplo, as preferência dos alunos ou o fato de alocar um aluno em uma opção com satisfação 0 não mudar o valor global da solução.

Este item envolve modificar a ordem que as soluções são analisadas de maneira a tentar encontrar primeiro as soluções de maior satisfação global. Combinada com o item anterior, está estratégia pode diminuir consideravelmente o tempo de execução.

Será obrigatório implementar este item em cima do branch and bound.

### Parte 2: Paralelismo

Cada item da parte anterior deverá ser paralelizado usando OpenMP. Isto deverá ser feito da maneira mais escalável possível: o tempo de execução do programa deverá diminuir conforme o número de cores aumenta.

Este item é o principal da entrega, mas ele se apoia inteiramente em cima dos itens da primeira entrega.

!!! warning
    Os projetos da disciplina consideram que não é razoável tentar paralelizar um problema que vocês não sejam capazes de resolver sequencialmente.


## Avaliação

A entrega será dividida em duas partes:

1. entrega do algoritmo sequencial em *C++*, comparando-o com a implementação em Python (30/03)
1. entrega final (antes da prova)

Na primeira entrega são esperados os seguintes itens:

- [ ] [Requisitos básicos](checklist.md) cumpridos
- [ ] Relatório comparando desempenho Python vs C++

A entrega final será dividida em duas partes: estratégias de paralelização (55%) e relatório de desempenho (45%).

!!! warning "Requisitos básicos de projeto"
    Caso os [requisitos básicos de projeto](checklist.md) não sejam cumpridos sua nota será limitada a `D`.

### Estratégias de paralelização

Cada funcionalidade do projeto corretamente implementada de maneira sequencia **E** paralela corresponde a um conceito neste item.

* **Conceito D**: implementou a busca local e mostrou ganhos de desempenho proporcionais ao número de processadores disponíveis. Você deverá produzir executáveis nomeados `busca_local_seq` e `busca_local_par`.
* **Conceito C**: implementou o algoritmo recursivo ingênuo, mostrando ganhos de desempenho proporcionais ao número de processadores disponíveis. Você deverá produzir executáveis nomeados `busca_exaustiva_seq` e `busca_exaustiva_par`.
* **Conceito B**: implementou o *Branch and Bound* usando a função de *bound*  mostrada no enunciado e compartilhando a melhor solução entre as threads. Você deverá produzir executáveis nomeados `branch_bound_seq` e `branch_bound_par`.
* **Conceito B+**: implementou uma heurística de busca no algoritmo sequencial e no paralelo e mostrou que há grandes ganhos de desempenho em seu uso. Você deverá produzir executáveis nomeados `branch_bound_heuristico_seq` e `branch_bound_heuristico_par`.
* **Conceito A+**: implementou uma solução híbrida, com uma thread rodando busca local e as restantes executando o *Branch and Bound*. Você deverá produzir executáveis nomeados `branch_bound_hibrido_seq` e `branch_bound_hibrido_par`.

Os conceitos acima estão apresentados em ordem de dificuldade. Em geral, o programa de um conceito se apoia nas implementações dos conceitos anteriores.

### Relatório

Você deverá produzir um relatório de desempenho seguindo os moldes do Projeto 1. Ele será avaliado de acordo com a [rubrica de relatórios da disciplina](rubrica.ods)

Os testes de seu projeto poderão ser limitados por tempo. Você pode limitar seu programa para rodar até no máximo 20 minutos. O objetivo é rodar entradas o maiores possível dentre

