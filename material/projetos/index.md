# Travelling Sales Person (TSP)

Nosso projeto do semestre será o [**Caixeiro Viajante**](https://en.wikipedia.org/wiki/Travelling_salesman_problem). Neste problema recebemos uma lista de cidades representadas por suas coordenadas $(x_i, y_i)_{i=0}^N$ e temos como objetivo encontrar o o caminho fechado que

1. visite todas as cidades
2. tenha o menor comprimento possível

Note que podemos começar de qualquer cidade, já que nosso caminho é fechado. Ou seja, ele começa e termina no mesmo lugar.

Este é um problema **muito** estudado em otimização e existem vários métodos que proporcionam soluções muito boas em pouquíssimo tempo. Por exemplo, o Gif abaixo usa uma técnica de otimização chamada [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing).

![Exemplo de solução do TSP](https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Travelling_salesman_problem_solved_with_simulated_annealing.gif/220px-Travelling_salesman_problem_solved_with_simulated_annealing.gif)

É fácil de perceber que este problema ocorre frequentemente em logísitca, seja na forma que apresentamos ou em sua generalização, o [Roteamento de Veículos](https://en.wikipedia.org/wiki/Vehicle_routing_problem).

## Por que escolhemos esse problema?

O *TSP* é um problema de otimização dos mais difíceis ([NP-hard](https://en.wikipedia.org/wiki/NP-hardness)). Não existe algoritmo polinomial que encontre a melhor solução **nem** algoritmo polinomial que cheque se uma solução é a melhor.

Entramos então em duas áreas:

1. encontrar a melhor solução o mais rápido possível
2. usar heurísticas para encontrar uma solução boa o mais rápido possível

Trabalharemos com ambas as ideias durante a disciplina, focando tanto em implementações sequenciais como paralelas.

## Avaliação e formatos de entrada e saída

O projeto é individual e será corrigido usando uma série de scripts de correção automatizada.

**Entrada**
```
N
x1 y1
....
xN yN
```

* `N` é o número de cidades
* `xi yi` são as coordenadas de cada cidade

**Saída**
```
L O
c1 .... cN
```

* `L` é o comprimento do *tour*
* `O` indica se o tour é ótimo (1) ou não (0)
* `c1 ... cN` é a sequência de cidades visitadas para chegar no tour de comprimento `L`

!!! warning "Importante"
    1. Os formatos de entrada e saída deverão ser respeitados de maneira estrita
    2. Algumas entregas pedirão também informações mostradas na saída de erros `std:cerr`
    3. Os scripts de correção serão atualizados durante o semestre.
    4.

## Entregas e Datas importantes

O projeto de vocês deverá ser entregue via um repositório Git privado criado especialmente para este fim. Ele será criado a partir das respostas do formulário a seguir.

<iframe width="640px" height= "480px" src= "https://forms.office.com/Pages/ResponsePage.aspx?id=wKZwY5B7CUe9blnCjt6DO-vbzw2O33BIuQfhB7kkTWxUNkdDWTJBVkJHTjZDVFA3Njc1MlQ4WldBOCQlQCN0PWcu&embed=true" frameborder= "0" marginwidth= "0" marginheight= "0" style= "border: none; max-width:100%; max-height:100vh" allowfullscreen webkitallowfullscreen mozallowfullscreen msallowfullscreen> </iframe>

Cada aluno irá adicionar sua solução ao seu repositório, que já irá conter os arquivos de correção automática. A entrega deverá cumprir todos os itens do [Checklist de projeto](checklist.md).

As etapas do projeto serão disponibilizadas depois de discussões em sala de aula e estarão conectadas com uma aula específica.

* Atividade 1 - [heurística da cidade mais próxima](heuristica): **Entrega 18/09**
* Atividade 2 - [busca local - troca de ordem](busca-local): **Entrega 30/09**
* Atividade 3 - [busca exaustiva](busca-exaustiva): **Entrega 13/10**
* Atividade 4 - [implementações sequenciais eficientes](desempenho-sequencial): **Entrega 04/11**


## Verificação de resultados

2. Você receberá um convite para repositório. Todas as atividades serão disponibilizadas neste repositório e suas soluções devem ser adicionadas nos arquivos correspondentes.
3. O corretor automático depende do pacote `grading-tools`, que deverá ser instalado como abaixo.

```shell
$> python3.8 -m pip install --user git+https://github.com/igordsm/grading-tools
```

??? tip "Python 3.8 no Ubuntu"
    Se seu `python3` é uma versão inferior ao 3.8, você pode instalá-lo com os pacotes abaixo:

    ```
    python3.8 python3.8-dev
    ```

    A partir daí poderá seguir normalmente as instruções desta página.

4. Com isso configurado, é só compilar seu programa e rodar `python3.8 corretor.py executavel`.
5. Para baixar os novos exercícios é só rodar `git pull`.
6. Os exercícios serão entregues criando um commit com sua resposta e dando `git push`.

!!! warning
    Fique atento a atualizações no seu repositório de projeto. Atualizações no corretor serão feitas ao longo do semestre, assim como serão disponibilizados novos arquivos de entrada/saída para cada parte a ser implementada.


