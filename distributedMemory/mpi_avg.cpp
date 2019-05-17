// Calcula a media de valores em um array com MPI_Scatter e MPI_Gather
// Author: Wes Kendall
// Copyright 2012 www.mpitutorial.com
// Atualizado: 15 de Maio de 2019 (Luciano Soares)

#include <iostream>
#include <mpi.h>
#include <assert.h>

// Cria um array com numeros aleatorios
float *create_rand_nums(int num_elements) {
  float *rand_nums = new float[num_elements];
  assert(rand_nums != NULL);
  for (unsigned int i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand()/(float)RAND_MAX);
  }
  return rand_nums;
}

// Calcula a media do array
float compute_avg(float *array, int num_elements) {
  float sum = 0.f;
  int i;
  for (i = 0; i < num_elements; i++) {
    sum += array[i];
  }
  return sum / num_elements;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Uso: ./avg numero_de_elementos_por_processo\n";
    exit(1);
  }

  int num_elements_per_proc = atoi(argv[1]);

  MPI_Init(&argc, &argv);
  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size; MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Cria um array de numeros aleatorios
  float *rand_nums = NULL;
  if (world_rank == 0) {
    rand_nums = create_rand_nums(num_elements_per_proc * world_size);
  }

  // Criar buffer para guardar parte do array
  float *sub_rand_nums = new float[num_elements_per_proc];
  assert(sub_rand_nums != NULL);

  // Enviar os valores aleatorios do processo 0 para todos os outros
  MPI_Scatter(rand_nums, num_elements_per_proc, MPI_FLOAT, sub_rand_nums,
              num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Calcula a média do conjunto de dados recebido
  float sub_avg = compute_avg(sub_rand_nums, num_elements_per_proc);

  // Envia de volta as médias parciais
  float *sub_avgs = NULL;
  if (world_rank == 0) {
    sub_avgs = new float[world_size];
    assert(sub_avgs != NULL);
  }
  MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

  // Realiza o calculo final da media
  if (world_rank == 0) {
    float avg = compute_avg(sub_avgs, world_size);
    std::cout << "Media de todos os elementos = " << avg << std::endl;
    // Calcula a media com os valores originais do array para comparacao
    float original_data_avg =
      compute_avg(rand_nums, num_elements_per_proc * world_size);
    std::cout << "Media calculada com os valroes originais = " << original_data_avg << std::endl;
  }

  // limpa dados
  if (world_rank == 0) {
    delete[] rand_nums;
    delete[] sub_avgs;
  }
  delete[] sub_rand_nums;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}