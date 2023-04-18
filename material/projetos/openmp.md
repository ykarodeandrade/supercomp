# Paralelismo com OpenMP

Até agora experimentamos heurísticas que buscaram resolver o nosso problema em um tempo razoável, sem garantias de otimalidade. É chegado o momento de incorporar o paralelismo de tarefas em nossas alternativas de resolução.

Para isso, você deve modificar a versão **exaustiva** de sua implementação. Você pode fazer uso da diretiva `#pragma omp parallel for` para distribuir as iterações de um loop entre as threads disponíveis. Dentro do loop, você pode fazer a verificação de cada filme e, caso ele esteja dentro das restrições de horário e categoria, incrementar uma variável compartilhada `count`. Observe que por ser uma variável compartilhada, você precisa preservar essa região crítica entre as threads. 

Vale ressaltar que o uso do OpenMP não necessariamente irá garantir um desempenho melhor, pois a paralelização tem um overhead que pode acabar diminuindo a performance do programa em alguns casos. É importante fazer testes para verificar se a utilização do OpenMP é realmente benéfica para o problema em questão.
