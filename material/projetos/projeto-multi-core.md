# Multi-core usando OpenMP

O primeiro projeto consiste em implementar uma solução multi-core para o problema da [Alocação de alunos do PFE](projeto-pfe.md). Para facilitar sua ele será dividido em duas partes.

Os algoritmos serão construídos em cima da [primeira parte](projeto-algoritmos.md), que tratou de estratégias eficientes de resolução do problema.

## Avaliação

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

Os conceitos acima estão apresentados em ordem de dificuldade. Em geral, a implementação de um conceito se apoia nas implementações dos conceitos anteriores.

### Relatório

Você deverá produzir um relatório de desempenho seguindo os moldes do Projeto 1. Ele será avaliado de acordo com a [rubrica de relatórios da disciplina](rubrica.ods)

Os testes de seu projeto poderão ser limitados por tempo. Você pode limitar seu programa para rodar até no máximo 20 minutos. O objetivo é rodar entradas o maiores possível dentre

