# Paralelismo com GPU

Esta etapa do projeto consiste em resolver nosso problema por meio da biblioteca Thrust. Vamos começar revendo a formalização de nosso problema:

Entrada:

Um inteiro N representando o número de filmes disponíveis para assistir.
Três vetores H, F e C de tamanho N, onde H[i] é a hora de início, F[i] é a hora de término e C[i] é a categoria do i-ésimo filme.
Um inteiro M representando o número de categorias.
Um vetor L de tamanho M, onde L[j] é o número máximo de filmes que podem ser assistidos na categoria j.

Saída:

Um inteiro representando o número máximo de filmes que podem ser assistidos de acordo com as restrições de horários e número máximo por categoria.


Para resolver esse problema utilizando a biblioteca thrust, podemos utilizar um algoritmo de programação dinâmica para construir a solução de forma eficiente. O algoritmo consiste em criar uma matriz dp de tamanho (N+1) x (M+1) para armazenar o número máximo de filmes que podem ser assistidos até o filme i e a categoria j.

Segue abaixo um pseudo-código (incompleto) para resolver o problema


````C++
// Carregar os dados do arquivo de entrada na memória da GPU
thrust::device_vector<int> start_times(N);
thrust::device_vector<int> end_times(N);
thrust::device_vector<int> categories(N);

// Ler os dados do arquivo de entrada
// ...

// Criar a matriz de programação dinâmica
thrust::device_vector<int> dp((N+1) * (M+1), 0);

// Inicializar a primeira linha da matriz com zeros
thrust::fill(dp.begin(), dp.begin() + M + 1, 0);

// Preencher a matriz com as soluções para subproblemas menores
for (int i = 1; i <= N; i++) {
  for (int j = 1; j <= M; j++) {
    // Encontrar o número máximo de filmes que podem ser assistidos até o filme i e categoria j
    int max_count = 0;
    for (int k = 0; k < i; k++) {
      if (categories[k] == j && end_times[k] <= start_times[i] && dp[(k*(M+1)) + j-1] + 1 <= L[j-1]) {
        max_count = max(max_count, dp[(k*(M+1)) + j-1] + 1);
      } else {
        max_count = max(max_count, dp[(k*(M+1)) + j]);
      }
    }
    dp[(i*(M+1)) + j] = max_count;
  }
}

// Encontrar o número máximo de filmes que podem ser assistidos
int max_count = 0;
for (int j = 1; j <= M; j++) {
  max_count = max(max_count, dp[(N*(M+1)) + j]);
}

// Escrever o resultado no arquivo de saída
// ...



````


A ideia do algoritmo é criar uma matriz dp de tamanho (N+1) x (M+1) para armazenar o número máximo de filmes que podem ser assistidos até o filme i e a categoria j. O algoritmo preenche a matriz com as soluções para subproblemas menores, até chegar na solução do problema original.

Para cada célula (i,j) da matriz dp, o algoritmo verifica se é possível adicionar o filme i à categoria j, respeitando as restrições de horário e limite máximo de filmes por categoria. Em seguida, o algoritmo verifica se é melhor adicionar o filme i à categoria j ou manter a solução anterior sem o filme i. O número máximo de filmes que podem ser assistidos é o valor da célula (N, j) da matriz dp, onde j é a categoria que maximiza o número de filmes assistidos.

Sua tarefa é realizar essa implementação em C++ com a Thrust e comparar o desempenho frente as demais implementações. 