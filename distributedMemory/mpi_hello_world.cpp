// Exemplo de programa Hello World para MPI
// Coding style: Luciano Soares
// Atualizado em 9/5/2019

#include <mpi.h>
#include <iostream>

int main() {
   int processes;
   int size;
   int rank;
   char processor[MPI_MAX_PROCESSOR_NAME];
   int name_len;

   MPI_Init(NULL, NULL);	//inicia o MPI

   MPI_Comm_size(MPI_COMM_WORLD, &size); // recupera número de processos

   MPI_Comm_rank(MPI_COMM_WORLD, &rank); // recupera o indice (rank) do processo

   MPI_Get_processor_name(processor, &name_len); // recuper nome do processor (host) e seu tamanho

   std::cout << "host: " << processor << "(rank " << rank << " de " <<  size << ")\n";

   MPI_Finalize(); // encerra o MPI
}
