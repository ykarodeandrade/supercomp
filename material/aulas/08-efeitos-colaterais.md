# 08 - Efeitos colaterais

Escrever código sem efeitos colaterais pode render ganhos de desempenho significativos ao paralelizar programas.

!!! note "Conceito"
    Uma função tem efeitos colaterais quando ela modifica dados que não são nem passados via argumentos nem retornados
    pela função. São exemplos desse tipo de dados:
    
    - variáveis globais
    - variáveis alocadas via `malloc`
    - variáveis apontadas por ponteiros
    
    Uma função **sem efeitos colaterais** recebe todos os seus argumentos por cópia, os processa e devolve seu resultado exclusivamente via retorno da função. Ou seja, todas as outras variáveis do programa mantém o mesmo valor que tinham antes da chamada dessa função.
    
Pela definição acima, notamos que uma função **sem efeitos colaterais** pode ser chamada por várias threads ao mesmo tempo. Claramente isto é uma vantagem, já que isto pode facilitar muito a paralelização de código. 

# Parte 0: analizando o código existente

Nesta parte do roteiro iremos analisar o código exemplo, testar uma paralelização ingênua e identificar seus possíveis problemas de paralelização.

!!! question short
    Considerando somente o arquivo *pi_mc.c*, existe código com  efeitos colaterais? 
    
!!! example
    Faça uma paralelização ingênua deste código.

!!! example 
    Teste a paralelização ingênua do exercício anterior. Ela retorna os mesmos resultados em todas execuções? Se não, comente por que isto é um problema.

Dado que não encontramos problemas no arquivo *pi_mc.c*, vamos olhar então os arquivos *random.c/h*. 

!!! question short
    Existe código com efeitos colaterais? Liste as funções encontradas.

!!! question short
    Voltando para *pi_mc.c*, onde são chamadas as funções identificadas acima?
    
!!! question short
    Agora que você está familiarizado com todo o código, explique por que os resultados são diferentes quando rodamos o código ingenuamente paralelo.

Só prossiga após validar as respostas do item anterior com o professor ou com um colega que já tenha finalizado esta parte.

# Parte 3 - exclusão mútua

Identificamos na parte anterior que a função `drandom` possui efeitos colaterais e estes efeitos colaterais estão atrapalhando a paralelização do código. Esta implementação é parecida com as funções do cabeçalho `<random>`: temos um estado do gerador de números que é passado para toda função que faz sorteios. Neste caso, o resultado do próximo número sorteado depende dos valores globais `MULTIPLIER`, `ADDEND`, `PMOD` e da variável estática `random_last`.

!!! example 
    Modifique o código para que ele use as funções de geração de números aleatórios usando `<random>`. Salve em *pi_mc_random.cpp*.

Na aula *06* usamos `omp critical` para criar seções de exclusão mútua em que somente uma das threads rode por vez. No nosso caso o problema ocorre com a geração de números aleatórios: ao gerar um número o estado interno do gerador se modifica. Logo, tem um problema de acessos concorrentes a mesma variável.

!!! question short
    Partindo do código sequencial, quais linhas precisariam ser protegidas de acessos concorrentes?

!!! example 
    Utilize `omp for` e  `omp critical` para paralelizar o código em *pi_mc_random.cpp*. 
    
!!! question short
    Avalie seu código em termos de desempenho obtido e facilidade de programação 

# Parte 2: paralelização das distâncias

Na parte anterior eliminamos acessos concorrentes ao gerador de números aleatórios. Isso seria equivalente a fazer todos os sorteios **antes** e depois fazer os cálculos das distâncias. Ou seja, os cálculos de distância são independentes desde que os sorteios aleatórios já tenham sido feitos. 

!!! example 
    Modifique *pi_mc_random.cpp* para que o sorteio dos pontos seja feito em um vetor **antes** do `for` que faz os cálculos de distâncias. 

!!! question medium 
    É possível paralelizar o `for` que faz o sorteio dos números? E o que faz o cálculo das distâncias? 

!!! example 
    Com base em suas respostas do item acima, paralize o que for possível e salve em um arquivo *pi_mc_par1.cpp*

!!! question short
    Houve ganho de desempenho? Compare com o programa original.

Esta estratégia é muito comum em casos em que a tarefa de interesse pode ser decomposta em uma parte inerentemente sequencial e uma que pode ser paralelizada mas que depende dos resultados da parte sequencial. Se a parte paralelizável for custosa essa estratégia pode trazer ganhos mesmo que o programa não seja inteiramente paralelizável. 

# Parte 3: partes independentes

A segunda estratégia que usaremos é criar um gerador de números aleatórios para cada thread. Ou seja, cada thread precisará:

1. criar um gerador de números aleatórios próprio
1. acumular os resultados na mesma variável de somar pontos. 

Note que esta estratégia não é equivalente ao programa original: cada thread está seguindo uma sequência diferente de números aleatórios. 

!!! example 
    Reorganize seu código (partindo de *pi_mc_random.cpp*) para que cada thread crie seu próprio gerador de números aleatórios e faça 1/4 das iterações originais. Salve seu trabalho em *pi_mc_par2.c*

!!! question short 
    Compare o desempenho com o programa original. 

# Parte 4 - comparação de desempenho final

Compare o desempenho das duas abordagens de paralelização. 

!!! question short
    Qual é mais rápida?
    
!!! question short
    Qual é mais fácil de ser entendida?
    
