# Busca local

<style>
.admonition p:nth-child(2) {
    font-size: 18pt;
}
</style>

Nossa busca local consistirá na seguinte estratégia:

!!! quote
    Trocar a ordem de visita de duas cidades

Ou seja, se for possível inverter a ordem de visitação de duas cidades e isso melhorar a solução então faça a troca. Só pare quando isso não for mais possível.

Para uniformizar nossas soluções vamos adotar a seguinte estratégia de desempates:

!!! quote
    Se houver mais de uma troca, escolha a com primeira cidade de menor índice. Se houve empate escolha a com a segunda cidade com menor índice.

## Geração de números aleatórios

* Seu programa deverá retornar o mesmo resultado em todas execuções. 
* Deverá ser possível configurar a *seed* usada no programa usando a [variável de ambiente](https://en.wikipedia.org/wiki/Environment_variable) `SEED`.  

> `$> SEED=20 ./busca-local < in.txt

* Se não for passada via variável de ambiente, assuma `SEED=10`.
* O algoritmo deverá gerar `10N` soluções e retornar a melhor delas. 

## Correção automática

Além da resposta correta, seu algortimo deverá também mostrar informações na saída de erros (`std::cerr`). Estas informações serão usadas para checar corretude de sua implementação.

!!! quote
    A cada solução gerada pela busca local (após o processo de trocas), seu programa deverá mostra em uma linha da saída de erros a seguinte linha:

    > `local: size c1 c2 ... cN`

    * `size` é o tamanho do tour encontrado
    * `c1 .... cN` é o tour que tem tamanho `size`

Será verificado na correção automática se as soluções locais produzidas obedecem à regra do início da seção. 