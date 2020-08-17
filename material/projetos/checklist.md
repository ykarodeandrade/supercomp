# Checklist de projeto 

Alguns requisitos de projeto básicos serão verificados em todas as entregas. O objetivo aqui é evitar que uma evidência importante de aprendizado deixe de ser enviada e prejudique a avaliação. Não cumprir algum desses requisitos implica em reentrega valendo **D**. 

- [ ] Script de compilação (CMake, Makefile, bash script). 
- [ ] Relatório feito em Jupyter Notebook ou PWeave. 
    - [ ] Versão em PDF (ou HTML) do relatório
    - [ ] Instruções para replicar os testes realizados. Se isto estiver incluso no relatório executável basta indicar no texto. 
    - [ ] Seção explicando onde está e como usar o script de compilação. 
    
## Requisitos de qualidade

Além dos requisitos acima, que são obrigatórios, os seguintes itens procuram medir a **qualidade** da implementação que vocês desenvolveram. Assim como explicado na aula 01, eles não reprovam, mas diminuem a nota de projeto. Cada item que não for cumprido implica em desconto de 1,0 na nota de projeto. 

- [ ] Programa compila sem warnings quando compilado com `g++ -O3 -Wall -pedantic -std=c++11`;
- [ ] Programa não tem warnings detectados pelo `clang-tidy`. Veja [este vídeo](https://www.youtube.com/watch?v=pXk6xIFWzv4) para um breve tutorial de uso;
    - usar os checks `read*,performance*,hicpp*,modern*,-modernize-use-trailing-return-type`. 
- [ ] Não há repetição desnecessária de código;
    - Implementaremos várias soluções para o mesmo problema. É importante compartilhar o máximo de código possível entre todas as implementações;
- [ ] Utilizar `struct` ou `class` para agregar grupos de variáveis que sempre são usadas junto (Exemplo: todos os dados de entrada do programa);
- [ ] Utilizar `typdef` para renomear tipos com nomes grandes. (Exemplo seria um `vector` de um `struct` ou `pair`);
- [ ] Utilizar corretamente os recursos de C++. Em geral o que for apresentado em aula e não conflitar com os itens acima é OK. Em geral código copiado da internet não passa nesse quesito. Na dúvida pergunte. 