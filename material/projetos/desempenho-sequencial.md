# Desempenho sequencial

Até o momento não nos preocupamos com a eficiência de nossos programas. Checamos sua corretude em termos dos resultados obtidos, mas não da eficiência de sua implementação. Agora vamos focar nisso.

Como visto na [aula 11](/aulas/11-introducao-paralelismo/), paralelizar um algoritmo ruim traz ganhos de desempenho muito limitados. É só quando aliamos algoritmo, implementação eficiente e paralelismo que desenvolvemos uma solução de alto desempenho. 

## Checagem automática

Continuaremos usando o `corretor.pyc`, que agora irá testar a eficiência dos algoritmos de busca local e exaustiva. No primeiro caso, vocês irão precisar implementar corretamente um algoritmo que avalie somente a diferença causada pela troca sem reavaliar o tour inteiro. No segundo caso, é impossível passar em todos os testes sem implementar algum tipo de branch-and-bound/best-first search.

Os testes de desempenho não irão considerar a saída de erros, verificando somente o tempo de execução e a validade da saída final. Para desabilitar isso vocês devem modificar o programa de vocês para ler o estado de uma variável de ambiente chamada `DEBUG`. 

* Se `DEBUG=1` seu programa deverá mostrar as mensagens na saída de erros. 
* Caso `DEBUG` não esteja definido ou seu valor não seja `1`, seu programa não deverá mostrar as mensagens na saída de erros. Note que o resultado final continuará o mesmo, estamos desligando apenas as 

!!! danger
    Dado que estamos permitindo que o programa cheque se está rodando no modo medição de desempenho, se for feito algo além do desligamento dos `print` seu trabalho automaticamente falha nesta atividade. 
