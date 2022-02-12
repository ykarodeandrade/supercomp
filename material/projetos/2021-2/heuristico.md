# Solução heurística

Um dos melhores estratégias para resolução do problema min-set-cover é a estratégia gulosa. O algoritmo guloso encontra uma solução para o problema de cobertura de conjunto escolhendo iterativamente um conjunto que cobre o maior número possível de variáveis descobertas restantes.

Sua tarefa: implemente a estratégia gulosa para o problema do min-set-cover. A cada iteração, **o algoritmo deve selecionar o subconjunto de F que irá cobrir o maior número de elementos de U que estavam descobertos**.


Veja abaixo um pseudo-código da estratégia gulosa que você deve implementar.

![greedy](https://i.stack.imgur.com/v55Gn.png)



 Faça testes para diversos tipos de entradas, e foque principalmente em uma grande quantidade de elementos e subconjuntos (n > 250).

Você deve entregar, além de código-fonte e todas as entradas e saídas geradas para o seu programa, um arquivo contendo o resultado do programa `verify` ( que você implementou ) e comentar sobre o número de vezes em que você conseguiu encontrar uma solução para o problema. Comente também sobre o tempo de execução de sua implementação. 

Para a entrega, usaremos o site **codePost**, você recebeu na sala de aula o link para criar sua conta. A submissão será feita unicamente por ele. Caso tenha alguma dúvida, entre em contato.
