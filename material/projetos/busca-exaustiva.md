<style>
.admonition p:nth-child(2) {
    font-size: 18pt;
}
</style>

# Busca exaustiva

Vamos agora explorar a criação de um algoritmo que encontra **o tour com a menor distância possível**. Isto significa que teremos que listar todas as possibilidades e retornar a melhor possível.

Nesta primeira etapa **não** iremos implementar algoritmos de busca exaustiva eficientes, como *Branch and Bound* e *Best-first search*. O requisito é conseguir implementar um algoritmo recursivo simples para resolver o problema.

## Correção automática

Além da resposta correta, seu algortimo deverá também mostrar informações na saída de erros (`std::cerr`). Estas informações serão usadas para checar corretude de sua implementação. Para auxiliar na correção você deverá mostrar na saída de erros os número de vezes que seu programa encontrou uma solução válida (que é igual também ao número de vezes que é feita a comparação com a melhor solução até o momento). Formate sua saída como mostrado abaixo

```
num_leaf XXX
```

