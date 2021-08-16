# Solução heurística

Um dos melhores estratégias para resolução do problema min-set-cover é a estratégia gulosa. O algoritmo guloso encontra uma solução para o problema de cobertura de conjunto escolhendo iterativamente um conjunto que cobre o maior número possível de variáveis descobertas restantes.

Sua tarefa: implemente a estratégia gulosa para o problema do min-set-cover. A cada iteração, o algoritmo deve selecionar o subconjunto de F que irá cobrir o maior número de elementos de U que estavam descobertos. Faça testes para diversos tipos de entradas, e foque principalmente em uma grande quantidade de elementos e subconjuntos (n > 250)