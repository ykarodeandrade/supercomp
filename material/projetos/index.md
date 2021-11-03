# Min-set-cover

O problema `min-set-cover` é um dos mais clássicos problemas de otimização combinatória. Uma instância do min-set-cover consiste em um universo `U` de objetos e uma coleção `F` de subconjuntos de `U`. O desafio é identificar uma menor sub-coleção `S` de `F` sujeita à restrição de que a união dos conjuntos em `S` deve cobrir o universo `U`. 

Por exemplo, suponha que uma cidade deseje selecionar um conjunto de espaços físicos para implantar uma rede de corpos de bombeiros de forma que cada casa esteja a menos de 5 km de pelo menos um corpo de bombeiros, e de forma que o número de corpos de bombeiros necessários seja minimizado. Nesse cenário, o conjunto de casas forma o universo `U` de itens que devem ser cobertos por qualquer solução; e a i-ésima localização possível para um corpo de bombeiros dá origem a um conjunto `Si` de residências que estão a 5 km dele. Supondo também que o conjunto total de locais possíveis seja suficientemente numeroso para que cada casa esteja a 5 km de um ou mais dos locais em consideração. A solução min-set-cover necessária é o menor conjunto de locais `S` de modo que a união desses sub-conjuntos cubra todas as residências do município. Ou seja, o que é necessário é um menor subconjunto `S` tal que cada casa em U apareça em pelo menos um elemento em `S`.

O problema min-set-cover, é NP-hard e, portanto, devemos fazer uso de técnicas aproximadas eficientes que possam encontrar uma boa solução.

Seu programa tomará como entrada a seguinte estrutura: a primeira linha contém dois números n e m. Onde `n` é `|U|` e `m` é o número de subconjuntos. Assuma `n ≤ 200` e `m ≤ 2^n`

```
10 5
5 7 8 9 10
5 7
1 3 6 10
4 6 7 9 10
1 2 8 10
```

A saída deve ser uma linha de inteiros `[1...m]`, contendo o número dos subconjuntos que formam a solução:

```
1 2 4 5
```

**Verificação automática**:

Conforme o número de entrada aumenta, ficará cada vez mais custoso verificar se a sua solução resolve o problema do min-set-cover. Faça um programa que recebe como entrada o arquivo de input original adicionado 

Exemplo:

```
./verify < solucao1.txt
```

onde `solucao1.txt` contém:

```
10 5
5 7 8 9 10
5 7
1 3 6 10
4 6 7 9 10
1 2 8 10
1 3 4 5
```

E tem como saída: `Cobertura atendida`.
<!-- 
Dados `M` objetos com valor `V[i], i=1..M` e `N` pessoas, desejamos dividir estes objetos de maneira mais igualitária possível. Como não é possível "quebrar" objetos, naturalmente alguns ficarão com objetos de maior valor que os outros. Nosso objetivo neste projeto é definir qual seria o *menor valor que uma pessoa deveria aceitar nesta partição*.

Para fazer isso vamos usar o seguinte procedimento: uma pessoa será responsável por fazer a partição dos objetos em `N` partes. Porém, ela deverá permitir que **todas as outras `N-1` pessoas escolham primeiro qual parte elas desejam**. Ou seja, a pessoa que fez a partição naturalmente ficará com a parte de menor valor. Portanto nosso objetivo será **maximizar** o valor da parte de **menor** valor. Chamaremos este valor de *MMS* e a atribuição que o gera de *parte 1-de-n*.

Vejamos um exemplo: separaremos 6 objetos para 3 pessoas. Os valores dos objetos são `{20, 11, 9, 13, 14, 37}`. Uma possível partição seria

```
{37}
{20, 11}
{14, 13, 9}
```

Com esta partição, o menor valor seria o do segundo grupo (31). Note que várias divisões são possíveis:

```
{37}
{20, 14}
{13, 11, 9}
```

Nesta outra partição o menor valor é o do terceiro grupo (33). Portanto, entre essas duas divisões a segunda é melhor, já que a pessoa que dividiu ganharia um valor maior.

Usaremos este problema na disciplina por uma razão bem simples: encontrar o *MMS* é uma tarefa *NP-difícil*. Ou seja, o melhor que podemos fazer neste caso para garantir a melhor solução é, no pior caso, testar todas as alocações possíveis. Claramente isso é lento, então é uma bom exemplo de aplicação de SuperComputação! -->
**Gerador de instâncias**

Neste [link](https://colab.research.google.com/drive/16MSMeqDoZF3zBFIQw_Yt7Y601v6MpxaW?usp=sharing) você encontra um código Python que pode ser utilizado para gerar instâncias para seus programa. Observe que não há garantia de que as instâncias geradas possuem solução.



## Técnicas estudadas e correção automática

Para cada técnica estudada em aula implementaremos versões básicas e avançadas. Também será necessário implementar versões paralelas em CPU e GPU. Veja abaixo as datas de entrega e descrições de cada técnica implementada. Em geral, o enunciado de uma parte é liberado após a data de entrega da parte anterior.

1. [Solução Heurística](heuristico) (15/09)
2. [Busca Local](busca-local)(22/09)
3. [Busca Exaustiva](busca-exaustiva)(27/09)
4. [Relatório Preliminar](relatorio-1) (02/11)
5. [Paralelismo Multicore](paralelismo-multicore) (15/11)
6. [Paralelismo GPU](paralelismo-gpu) (22/11)
7. [Relatório Final](relatorio-2) (06/12)

<!-- 1. [Solução Heurística](heuristico) (23/03)
2. [Busca local](busca-local) (09/04)
3. [Busca exaustiva](busca-exaustiva) (16/04)
4. [Relatório preliminar](relatorio-1) (30/04)
5. [Paralelismo MultiCore](paralelismo-multicore) (21/05) -->
<!-- 
Cada parte de implementação será conferida usando um script de correção checagem de resultados disponível no repositório de entregas do projeto, juntamente com instruções de uso. Registre seu usuário do github até **15/03** para ser convidado para seu repositório de entregas.

<iframe width="640px" height= "480px" src= "https://forms.office.com/Pages/ResponsePage.aspx?id=wKZwY5B7CUe9blnCjt6DO36bxJ3XetxChDUDKdweTOJURUNKWkFLSklHNk1RWlVBTUNHWEszVExOViQlQCN0PWcu&embed=true" frameborder= "0" marginwidth= "0" marginheight= "0" style= "border: none; max-width:100%; max-height:100vh" allowfullscreen webkitallowfullscreen mozallowfullscreen msallowfullscreen> </iframe>

### Instruções de correção

O corretor automático depende do pacote `grading-tools`, que deverá ser instalado como abaixo.

```shell
$> python3.8 -m pip install --user git+https://github.com/igordsm/grading-tools
```

??? tip "Python 3.8 no Ubuntu"
    Se seu `python3` é uma versão inferior ao 3.8, você pode instalá-lo com os pacotes abaixo:

    ```
    python3.8 python3.8-dev
    ```

    A partir daí poderá seguir normalmente as instruções desta página.

Com isso configurado, é só compilar seu programa e rodar `python3.8 corretor.py executavel`. Para baixar os novos exercícios é só rodar `git pull`. Os exercícios serão entregues criando um commit com sua resposta e dando `git push`.


!!! warning
    Fique atento a atualizações no seu repositório de projeto. Atualizações no corretor serão feitas ao longo do semestre, assim como serão disponibilizados novos arquivos de entrada/saída para cada parte a ser implementada. -->

## Avaliação

O projeto será avaliado usando rubricas para as entregas básicas. As rubricas de avaliação dos relatórios estarão descritas em suas páginas de entrega.

### Conceito D

Algum dos seguintes itens não foi entregue corretamente ou possui problemas sérios (no caso do relatório final).

1. Solução heurística
2. Busca local
3. Busca exaustiva
4. Busca local paralela (CPU)
5. Busca local paralela (GPU)
6. Relatório preliminar
7. Relatório final


### Conceito C

Todas as atividades abaixo foram validadas pelo corretor e (no caso do relatório final) alcançaram qualidade mínima exigida.

1. Solução heurística
2. Busca local
3. Busca exaustiva
4. Busca local paralela (CPU)
5. Busca local paralela (GPU)
6. Relatório preliminar
7. Relatório final

### Conceito C+

Além do já validado no conceito **C**, os relatórios entregues não tinham nenhum ponto **em desenvolvimento** ou **insatisfatório** na rubrica do relatório.

### Conceitos avançados

A partir do  conceito **C+** cada atividade avançada vale meio conceito. Elas serão listadas aqui conforme o semestre avança e serão testadas pela checagem de resultados disponível no repositório de entregas.
