// Programa calcula o desvio padrao dos elementos de um vetor com MPI_Reduce.
// Author: Wes Kendall
// Copyright 2013 www.mpitutorial.com
// Atualizado: 15 de Maio de 2019 (Luciano Soares)


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>

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

  // Faz a reducao das somas locais para calcular media
  float global_sum;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);
  float mean = global_sum / (num_elements_per_proc * world_size);

  // Calcula a diferenca ao quadrado da soma local e media
  float local_sq_diff = 0;
  for (unsigned int i = 0; i < num_elements_per_proc; i++) {
    local_sq_diff += (rand_nums[i] - mean) * (rand_nums[i] - mean);
  }

  // Faz a reducao final
  float global_sq_diff;
  MPI_Reduce(&local_sq_diff, &global_sq_diff, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // Desvio padrao
  if (world_rank == 0) {
    float stddev = sqrt(global_sq_diff /
                        (num_elements_per_proc * world_size));
    std::cout << "Mean - " << mean << ", Standard deviation = " << stddev << std::endl;
  }

  // Libera memoria
  delete[] rand_nums;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
