// Exemplo de programa de Broadcast em MPI
// Coding style: Luciano Soares
// Atualizado: 15 de Maio de 2019

#include <mpi.h>
#include <cstring>
#include <iostream>
#include <assert.h>

#define NUM_TESTS 10000

void linear_bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator) {
   int rank;
   int size;

   MPI_Comm_rank(communicator, &rank);
   MPI_Comm_size(communicator, &size);

   if (rank==root) {
      int i;
      for (i = 0; i < size; i++) {
         if (i!=rank) MPI_Send(data, count, datatype, i, 0, communicator);
      }
   } else {
      MPI_Recv(data, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);
   }
}

int main(int argc, char ** argv) {
   int rank;
   char data[100];
   char processor[MPI_MAX_PROCESSOR_NAME];
   int name_len;
   double starttime, endtime;
   double mpi_broadcast, meu_broadcast;
   unsigned int i;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Get_processor_name(processor, &name_len); // recupera nome do processor (host) e tamanho do texto

   // BROADCAST MPI
   starttime = MPI_Wtime();
   for(i=0;i<NUM_TESTS;i++) {
      if (rank == 0) {
         strcpy(data,"Ola do Master");
         MPI_Bcast(data, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
      } else {
         MPI_Bcast(data, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
         assert(strcmp(data,"Ola do Master") == 0);
      }
   }
   if (rank == 0) {
      mpi_broadcast = MPI_Wtime() - starttime;
      std::cout << "Tempo decorrido de " << mpi_broadcast << " segundos com Broadcast do MPI\n";
   }
   
   MPI_Barrier(MPI_COMM_WORLD);

   // BROADCAST MEU
   starttime = MPI_Wtime();
   for(i=0;i<NUM_TESTS;i++) {
      if (rank == 0) {
         strcpy(data,"Ola do Master");
         linear_bcast(data, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
      } else {
         linear_bcast(data, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
         assert(strcmp(data,"Ola do Master") == 0);
      }
   }   
   if (rank == 0) {
      meu_broadcast = MPI_Wtime() - starttime;
      std::cout << "Tempo decorrido de " << meu_broadcast << " segundos com meu Broadcast\n";
      std::cout << "Broadcast do MPI foi " << meu_broadcast/mpi_broadcast << " vezes mais rapido\n";
   }

   MPI_Finalize();
   return 0;
}
