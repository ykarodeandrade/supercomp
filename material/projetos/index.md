# Maratona de Filmes 

Você quer passar um final de semana assistindo ao máximo de filmes possível, mas há restrições quanto aos horários disponíveis e ao número de títulos que podem ser vistos em cada categoria (comédia, drama, ação, etc).

**Entrada**: Um inteiro N representando o número de filmes disponíveis para assistir e N trios de inteiros (H[i], F[i], C[i]), representando a hora de início, a hora de fim e a categoria do i-ésimo filme. Além disso, um inteiro M representando o número de categorias e uma lista de M inteiros representando o número máximo de filmes que podem ser assistidos em cada categoria.

**Saída**: Um inteiro representando o número máximo de filmes que podem ser assistidos de acordo com as restrições de horários e número máximo por categoria.


# Gerador de input para o projeto

O código abaixo gera um arquivo `input.txt` que deve ser usado como entrada para seu programa.

Como exemplo, considere o seguinte arquivo `input.txt` gerado:

```
10 4
1 3 1 2 
11 13 3
14 15 3
10 16 2
10 14 1
11 17 2
11 14 3
13 15 3
14 15 1
12 16 4
12 13 4
```

Como ler esse arquivo?

- a primeira linha indica que há 10 filmes a serem considerados e 4 categorias;
- a segunda linha indica qual o máximo de filmes que cada categoria pode ter;
- da terceira linha em diante você vai encontrar os `n` filmes, suas respectivas hora de início, hora de término e categoria pertencente. 

O código-fonte abaixo é apenas uma recomendação. Fique livre para adaptá-lo. De modo que os filmes tenham uma duração média de 3 horas (isso você pode variar e ver como fica a complexidade do problema), nós usamos a biblioteca Boost (https://www.boost.org) para gerar números aleatórios.

```cpp

#include <chrono>
#include <random>
#include <fstream>
#include <boost/random.hpp>

using namespace std;

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    ofstream inputFile;
    inputFile.open("input.txt");
    inputFile << n << " " << m << endl;

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);

    // Definindo distribuição de horas de início e fim com média de 3 horas de diferença
    double mean = 3.0; // média de diferença de 3 horas
    double sigma = mean / sqrt(3.0); // desvio padrão para manter a média de diferença
    boost::random::normal_distribution<double> distribution1(12.0, sigma);
    boost::random::normal_distribution<double> distribution2(12.0 + mean, sigma);

    uniform_int_distribution<int> distribution3(1, m);

    vector<int> maxFilmes(m); // Vetor para armazenar o máximo de filmes por categoria
    for (int i = 0; i < m; i++) {
        maxFilmes[i] = distribution3(generator); // Gerando o máximo de filmes para cada categoria
        inputFile << maxFilmes[i] << " "; // Escrevendo o valor no arquivo de entrada
    }
    inputFile << endl;

   for (int i = 0; i < n; i++) {
        int horaInicio = round(distribution1(generator));
        int horaFim = round(distribution2(generator));
        if (horaFim < horaInicio) {
            swap(horaFim, horaInicio);
        }
        int categoria = distribution3(generator);

        inputFile << horaInicio << " " << horaFim << " " << categoria << endl;
    }
    inputFile.close();
    return 0;
}

```

Para compilar um programa C++ que usa a biblioteca Boost no g++, você precisará seguir os seguintes passos:

1. Inclua os arquivos de cabeçalho da biblioteca Boost em seu código fonte. Por exemplo, se você quiser usar a biblioteca Boost.Random, inclua o arquivo de cabeçalho `<boost/random.hpp>` em seu código.

2. Compile seu código-fonte usando o g++ e informe ao compilador que você está usando a biblioteca Boost, usando a opção `-lboost_` seguida do nome da biblioteca. Por exemplo, se você estiver usando a biblioteca `Boost.Random`, use a opção `-lboost_random` ao compilar o seu código.

3. Certifique-se de que o compilador possa encontrar os arquivos de cabeçalho e bibliotecas da biblioteca Boost. Você pode fazer isso usando a opção `-I` para especificar o diretório de inclusão e a opção `-L` para especificar o diretório de linkagem. Por exemplo, se a biblioteca Boost estiver instalada em `/usr/local/Cellar/boost/1.75.0`, use as opções `-I/usr/local/Cellar/boost/1.75.0/include` e `-L/usr/local/Cellar/boost/1.75.0/lib` ao compilar o seu código.

Assim, um exemplo de comando para compilar um programa C++ que usa a biblioteca Boost seria:

```
g++ -I/usr/local/Cellar/boost/1.75.0/include -L/usr/local/Cellar/boost/1.75.0/lib -lboost_random meu_programa.cpp -o meu_programa
```
