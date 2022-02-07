# Solução Busca local

<!-- Vamos agora implementar uma estratégia de **Busca local** como fizemos na [aula 07](/aulas/07-busca-local). Nossa estratégia de busca local será composta por três passos:

!!! quote "Passo 1"
    Cada objeto é atribuído para uma pessoa aleatória.

!!! quote "Passo 2"
    1. Selecione a pessoa `P` com o menor valor.
    2. Passe por todos os outros objetos e verifique se esse objeto poderia ser doado para `P`.
        * **Um objeto pode ser doado se o valor total do doador tirando o objeto doado é maior que o valor total da pessoa `P`.**
        * Se for possível, faça a doação, calcule o novo *MMS*.

!!! quote "Passo 3"
    Repita o *Passo 2* até que não seja mais possível.

Para auxiliar o entendimento desta busca local, responda as seguintes perguntas.

!!! question short
    O *Passo 2* da nossa estratégia nunca diminui o *MMS*. Por que?

!!! question short
    Conseguimos saber de antemão quantas vezes repetiremos *Passo 2*?

## Variáveis de ambiente

Nosso programa será controlado por três variáveis de ambiente:

* `SEED` controla o seed usado em nosso gerador de números aleatórios. Se não for passado, assuma `SEED = 0`;
* `ITER` controla o número de vezes que repetimos a estratégia delineada acima. Se não for passado, assuma `ITER = 100000`;
* `DEBUG`: mostra informações auxiliares para ajudar a correção automática. Se não for passado, assuma `DEBUG = 0`;

Ao rodar com `DEBUG=1` seu programa deverá mostrar na saída de erros `cerr` uma linha para cada resultado final do processo de busca local. Ou seja, deverá mostrar `ITER` linhas no formato abaixo:

```
valor a1 ... aN
```

* `valor` contém o valor do *MMS* da solução
* `aI` contém a pessoa que possui o objeto `I`

Note que o formato dessa saída é diferente da saída final do programa, porém a informação representada é a mesma.

## Validação de resultados

A pasta `heuristica` do repositório de entregas contém arquivos de exemplo `in*.txt/out*.txt` com as entradas e saídas possíveis para esta parte do projeto. Use-os para validar seu programa junto com `corretor.py`.

!!! warning
    Esta validação não espera resultados idênticos aos das saídas de exemplo.

A validação testará se seu programa tem as seguintes propriedades:

* a solução final é ótima local. Ou seja, não é possível repetir o *Passo 2* e conseguir uma solução melhor
* a solução final é valida.
* as soluções de cada busca local mostradas quando `DEBUG=1` também são ótimos locais
* a solução final é a melhor entre todas as soluções calculadas
* soluções diferentes são geradas quando valores de `SEED` diferentes são passados
* são mostradas `ITER` linhas na saída de erros e cada linha corresponde a uma solução válida -->


A busca local consiste em uma metaheurística usada para resolver problemas de otimização computacionalmente difíceis. Esse tipo de algoritmo percorre o espaço de busca movendo-se iterativamente de uma solução candidata para outra, seguindo um caminho através da relação de vizinhança, até que uma solução considerada boa o suficiente seja encontrada ou um limite de tempo decorrido. Normalmente todo candidato possui mais de uma solução de vizinho e a escolha entre elas é feita com o auxílio de informações locais e experiência anterior.

A solução por busca local  tenta maximizar o número de elementos com o mínimo de subconjuntos possível. Precisamos capturar esse critério por meio de uma função de *fitness*. Uma maneira possível de fazer isso é construir uma função de *fitness* calculando o número de elementos capturados pelos subconjuntos de uma solução candidata e, em seguida, dividindo-o pelo número de subconjuntos que contém. Essa função de pontuação favorecerá as soluções que acumulam a maioria dos elementos do universo U com o mínimo de subconjuntos.


Para isso, implemente as seguintes alterações em seu projeto:

1. Gerar uma solução aleatória para o problema do min-set-cover;
2. Percorra novamente os conjuntos os elementos da sua solução e, de maneira randômica, troque até r (r entre 1 e 3) elementos da sua solução por subconjuntos que ficaram de fora da solução. 
3. Se a solução tiver melhor escore, mantenha ela. 

Para verificar o desempenho, construa um cenário com ao menos 200 elementos e 80 subconjuntos, de até 40 elementos cada.  Faça três variações desse cenário (elementos, subconjuntos, número de elementos em subconjuntos) e avalie o desempenho e a efetividade em encontrar uma solução ótima.


Para a entrega, usaremos o site **codePost**, você recebeu na sala de aula o link para criar sua conta. A submissão será feita unicamente por ele. Caso tenha alguma dúvida, entre em contato.
