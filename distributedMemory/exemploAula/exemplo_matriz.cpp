#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size; MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n;

    if(world_rank==0) {
        std::cout << "digite tamanho da matriz" << std::endl;
        std::cin >> n;
        MPI_Send(&n, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    double **M = new double*[n];
    for(int i=0;i<n;i++)
        M[i] = new double[n];

    if(world_rank==0) {
      for(int i=0;i<n;i++)
        MPI_Recv(M[i], n, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            std::cout << M[i][j] <<std::endl;
    } else {
      for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            M[i][j] = (i+1)*(j+1);
      for(int i=0;i<n;i++)
        MPI_Send(M[i], n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }

    delete[] M;
    MPI_Finalize();
    return 0;
}