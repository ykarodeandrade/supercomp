# Multi-core usando OpenMP

O primeiro projeto consiste em implementar uma solução multi-core para o problema da [Alocação de alunos do PFE](projeto-pfe.md). Para facilitar sua ele será dividido em duas partes.

Os algoritmos serão construídos em cima da [primeira parte](projeto-algoritmos.md), que tratou de estratégias eficientes de resolução do problema.

## Estratégias de paralelização

Cada funcionalidade do projeto corretamente implementada de maneira sequencial **E** paralela corresponde a um conceito neste item.

* **Conceito D**: implementou a busca local e mostrou ganhos de desempenho proporcionais ao número de processadores disponíveis. Você deverá produzir executáveis nomeados `busca_local_seq` e `busca_local_par`.
* **Conceito C**: implementou o algoritmo recursivo ingênuo, mostrando ganhos de desempenho proporcionais ao número de processadores disponíveis. Você deverá produzir executáveis nomeados `busca_exaustiva_seq` e `busca_exaustiva_par`.
* **Conceito A**: para alcançar este conceito seu programa deverá ter alto desempenho. Os tempos foram calibrados usando uma implementação do algoritmo heurístico sequencial, levando em conta que seu programa deveria cortar este tempo pela metade. Duas sugestões de paralelização são dadas abaixo. 
    * Algoritmo heurístico implementado 
    * Solução híbrida Branch and Bound + busca local.

## Validação

Cada conceito possui um validador individual que leva em conta particularidades de cada técnica implementada. 

* **Conceito D**: a ser disponibilizado.
* **Conceito C**: `validacao-paralelo-exaustivo.py`
* **Conceito A**: `validacao-bb-heur.py`
