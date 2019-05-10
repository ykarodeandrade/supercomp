#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int cmpfunc (const void * a, const void * b) {
	return ( *(int*)a - *(int*)b );
}

int main(int argc, char ** argv) {

	int rank, a[10], b[5], c[10];
	MPI_Init(&argc, &argv);
	double start = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		for(int f=0;f<10;f++) a[f]=rand()%1000;

		//for(int f= 0;f<10;f++) printf("%d\n",a[f]);

		MPI_Send(&a[5], 5, MPI_INT, 1, 0, MPI_COMM_WORLD);
		qsort(a, 5, sizeof(int), cmpfunc);
		MPI_Recv(b, 5, MPI_INT, 1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		/* Atividade: termine de ordenar os dados */
		int ca=0;
		int cb=0;

		for(int f=0;f<10;f++) {
			if( a[ca]<=b[cb] && ca<5 || cb >=5 ) c[f]=a[ca++];
			else c[f]=b[cb++];
		}
		for(int i=0;i<10;i++) {
			printf("%d\n",c[i]);
		}
		double end = MPI_Wtime();
		printf("tempo decorrido: %f\n",end - start);
	} else if (rank == 1) {
		MPI_Recv(b, 5, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		qsort(b, 5, sizeof(int), cmpfunc);
		MPI_Send(b, 5, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();

	return 0;

}

