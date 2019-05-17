// Programa calcula a media dos elementos de um vetor com MPI_Reduce.
// Author: Wes Kendall
// Copyright 2013 www.mpitutorial.com
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

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Uso: ./avg numero_de_elementos_por_processo\n";
    exit(1);
  }

  int num_elements_per_proc = atoi(argv[1]);

  MPI_Init(&argc, &argv);
  int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size; MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Cria um array de numeros aleatorios em todos os processos
  float *rand_nums = NULL;
  rand_nums = create_rand_nums(num_elements_per_proc);

  // Soma todos os numeros localmente
  float local_sum = 0;
  for (unsigned int i = 0; i < num_elements_per_proc; i++) {
    local_sum += rand_nums[i];
  }

  // Exibe os numeros aleatorios de cada processo
  std::cout << "Soma local por processo " << world_rank << " - " << local_sum << ", avg = " << local_sum / num_elements_per_proc << std::endl;

  // Faz a reducao das somas locais na soma global
  float global_sum;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // Exibe o resultado
  if (world_rank == 0) {
    std::cout << "Soma total = " << global_sum << ", avg = " << global_sum / (world_size * num_elements_per_proc) << std::endl;
  }

  // Limpa memoria
  delete[] rand_nums;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
