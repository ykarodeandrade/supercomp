# 03 - CMake e Debugging no VSCode

Até o momento todos nossos arquivos eram compilados em uma só linha usando `g++`. Conforme projetos ficam maiores e são divididos em vários módulos isto se torna inviável. Neste roteiro iremos usar o CMake, uma ferramenta de gerenciamento de projetos em C++. No fim, iremos configurar o VSCode para abrir projetos CMake.

!!! warning "Atenção"
    Usar ferramentas adequadas aumenta consideravelmente a produtividade e também facilita encontrar erros no código. A partir desta aula será obrigatório já ter debugado o programa antes de tirar dúvidas. 

## Gerenciamento de projetos com CMake

O [CMake](http://www.cmake.org) é uma das ferramentas mais usadas para gerenciar projetos C/C++. Outras ferramentas comumente usadas são [ninja](https://ninja-build.org/manual.html) e [meson](https://mesonbuild.com/). 

Em *CMake*, um projeto é definido por um arquivo nomeado *CMakeLists.txt*. Este arquivo contém definições como nome do projeto, quais dependências (bibliotecas externas e subprojetos) são usadas e o quais arquivos serão gerados pelo projeto. Cada arquivo gerado é chamado de *target* na nomenclatura usada pelo CMake e pode ser um executável ou uma biblioteca (estática ou dinâmica). 

Um arquivo *CMakeLists.txt* básico pode conter apenas três linhas (pasta [03-cmake](https://github.com/insper/supercomp/code/03-cmake)):

```
cmake_minimum_required(VERSION 3.10)
project (projeto_basico)
add_executable(hello hello.cpp) 
```

Este arquivo somente descreve o projeto. Para efetivamente compilarmos o programa precisamos passar pela fase de *configuração*, em que o *CMake* checa se todas as dependências foram encontradas e se os compiladores necessários estão instalados. Se tudo estiver em ordem podemos gerar um *Makefile* (para Linux) ou um projeto do Visual Studio (para Windows). 

Para fazer a configuração do projeto basta rodar o comando `cmake` mais o caminho para a pasta do projeto. Você só precisará refazer a configuração do projeto se modificar o arquivo *CMakeLists.txt*.  É boa prática fazer a compilação do código em uma pasta separada, como na sequência de comandos abaixo.

    mkdir build
    cd build
    cmake ..

Estes comandos devem ter gerado uma série de arquivos na pasta *build*, incluindo um *Makefile*. Para compilar o projeto basta rodar

    make
    
E um executável de nome *hello* deverá aparecer na pasta *build*. 

!!! note "Exercício"
    Crie um arquivo *CMakeLists.txt* para o projeto 0. Você deverá adicionar um *target* chamado `vector_ops`. 
    
!!! note "Exercício"
    Separe as funções matemáticas testadas em arquivos *.cpp/h* e use-os no seu projeto. A diretiva `add_executable` aceita vários arquivos *.cpp* para criar um executável. 
    
## Opções de compilação 

Além da criação de executáveis o CMake também permite adicionar opções de compilação específicas para cada *target* com a diretiva [`target_compile_options`](https://cmake.org/cmake/help/latest/command/target_compile_options.html). No contexto desta matéria isto será especialmente interessante pois o `g++` oferece flags para ativar otimizações que podem melhorar significativamente o desempenho de nosso programa. Portanto, podemos facilmente compilar o mesmo programa com e sem otimizações no mesmo projeto! O exemplo abaixo ativa a flag `O3` no *target* `hello` criado no exemplo anterior. 

    target_compile_options(hello PUBLIC -O3)

Podemos também usar `#define` para fazer compilação condicional do nosso código. Usar a [diretiva abaixo](https://cmake.org/cmake/help/latest/command/target_compile_definitions.html) equivale e colocar um `#define OPT` no topo de cada arquivo do target `hello`.

    target_compile_definitions(hello PUBLIC OPT)
    
!!! bug "Exercício" 
    Modifique seu *target* `vector_ops` para que ele use a opção de compilação `O2`.


!!! bug "Exercício" 
    Crie um novo *target* `vector_ops_no_opt` que compile os mesmos arquivos de `vector_ops` mas use a opção de compilação `O0`.
    

## Projeto 0 - Benchmarking e comparações de desempenho

Vamos agora fazer um pequeno resumo dos resultados dos experimentos. Este resumo deverá conter os seguintes itens e deverá ser entregue em formato *PDF*. 

- [ ] uma breve descrição do programa testado (um parágrafo); 
- [ ] os nomes dos executáveis testados e qual a diferença entre eles
- [ ] os tamanhos de entrada usados e a máquina usada para seus testes
- [ ] gráficos ilustrando a diferença de desempenho para cada função testada
- [ ] comentários sobre os resultados mostrados nos gráficos

### Dicas

1. Ferramentas como [PWeave](http://mpastell.com/pweave) ou Jupyter Notebook ajudam muito a criar textos que misturam código para criar gráficos, rodar programas automaticamente e interpretar sua saída. 
1. Tente automatizar o máximo possível a geração de gráficos de desempenho. Isso facilitará muito sua vida em projetos posteriores. 

## Extra 01 - Debugando seu projeto 

Agora iremos abrir nosso projeto um ambiente de desenvolvimento em vez de continuar trabalhando na linha de comando. Adotaremos o VSCode como ambiente padrão. 

O VSCode não vem com suporte por padrão a projetos CMake, mas a extensão [*Cmake Tools*](https://vector-of-bool.github.io/docs/vscode-cmake-tools/index.html) possui um bom suporte. Instale-a e carregue o projeto. A extensão possui uma documentação tanto para a etapa de [configuração e compilação](https://vector-of-bool.github.io/docs/vscode-cmake-tools/getting_started.html#configuring-your-project) quanto para [execução dos programas e debugging](https://vector-of-bool.github.io/docs/vscode-cmake-tools/debugging.html). 

Para executar os programas você pode usar a opção "Run in terminal" disponível no menu do botão direito de cada *target*. 

-------

Supondo que a etapa anterior funcionou, vamos agora rodar o nosso programa usando um debugger. Isto permite parar a execução no meio do programa e examinar o valor das variáveis. 

0. coloque um ou mais *breakpoints* no código. Basta clicar ao lado do número da linha. Um círculo vermelho indica que a execução será interrompida quando chegar nesta linha.
1. botão direito no target -> "Run with debugger".
2. quando o programa parar teremos as seguintes opções:
    * **Continue**: roda até o próximo *breakpoint*
    * **Step Over**: roda a linha atual e passa para a próxima.
    * **Step Into**: se a linha atual tem uma chamada de função, continua o debug dentro da função chamada.
    * **Step Out**: executa até o fim da função atual e para logo após o retorno.
3. Também podemos colocar o mouse em cima de cada variável para ver seu valor. O explorador de variáveis (que mostra todas as válidas na função atual) aparece no painel esquerdo, dentro da view *debug*.

!!! bug "Exercício"
    Abra sua atividade e rode-a com o debugger. Pause a execução no meio e verifique os tempos de execução. 
