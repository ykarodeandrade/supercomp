# Algoritmos para problemas NP-completo

Na primeira etapa do projeto nos concentraremos na resolução do [problema proposto](projeto-pfe.md) de maneira eficiente.
O arquivo `solucao-ingenua.py` contém uma solução simplista escrita em Python. Este programa sempre encontra a melhor solução, mas é **extremamente ingênuo**. Por isso, ele também é **extremamente lento**.

## Implementação em C++

Nosso primeiro passo será implementar esse mesmo programa em C++. O objetivo desta parte é compreender o efeito da linguagem de programação escolhida no desempenho da solução. Sua implementação deverá se chamar `busca_exaustiva_seq`.

Além de devolver a saída no [formato indicado no enunciado](projeto-pfe.md), seu programa também deverá mostrar informações de debug na saída de erros. Sempre que for encontrada uma solução melhor que a atual seu programa deverá mostrar uma linha como a abaixo na saída de erros:

```
Melhor: (sat) pa1 pa2 pa3 ... pa(n_alunos)
```

* `(sat)` é a satisfação da solução encontrada.
* `pa1` até `pa(n_alunos)` é o vetor contendo o projeto atribuído a cada aluno.

Seu programa deverá encontrar exatamente as mesmas soluções que o programa em Python, pois ele é uma tradução fiel do algoritmo utilizado e deveria percorrer as soluções possíveis na mesma ordem.

Para validar sua implementação deste item você deverá usar o script `code/projeto-validacao/validacao-exaustivo.py`. Este script recebe o seu executável e roda uma série de testes, verificando tanto a saída esperada quanto as informações de diagnóstico mostradas na saída de erros. As entradas usadas estão na pasta `entradas`.

<!--
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

Será obrigatório implementar este item em cima do branch and bound. -->

## Avaliação

* **Conceito D**: implementou o algoritmo exaustivo inocente em `C++`. O executável deverá ser nomeado `busca_exaustiva_seq`.
