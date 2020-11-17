# Paralelismo em GPU

A parte obrigatória final do projeto consiste em implementar a busca local em GPU. Os requisitos básicos serão os mesmos da [busca local](busca-local) e da parte de [desempenho sequencial](desempenho-sequencial)

* O programa retorna sempre os mesmos resultados.
* É possível configurar o *seed* do programa usando a variável de ambiente `SEED`

## Correção automática

O corretor compilará todos os arquivos *.cu* na pasta busca local em um executável `gpu`. Esse executável deverá produzir as mesmas informações de debug que a busca local sequencial se a variável de ambiente `DEBUG` tiver valor `1`. Se ela não existir assuma `DEBUG=0`.

O requisito básico para a implementação em GPU é sua velocidade. Por isso, serão aplicadas as mesmas checagens da implementação sequencial, mas agora com um tempo limite agressivo. A princípio, não seria possível passar nos testes usando somente a CPU.




