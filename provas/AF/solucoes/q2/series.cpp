#include <iostream>
#include <iomanip>
#include <math.h>
#include <mpi.h>

// Serie 1 = 1 + 1/2 + 1/4 + 1/8 + 1/6 + ...
double calc_serie_1(unsigned long int begin, unsigned long int end) {
    double calc = 0;
    for(unsigned long int l=begin;l<end;l++)
        calc += 1.0f / pow(2,l);
    return(calc);
}

// Serie 1 = 1 + 1/2 + 1/3 + 1/4 + 1/5 + ...
double calc_serie_2(unsigned long int begin, unsigned long int end) {
    double calc = 0;
    for(unsigned long int l=begin;l<end;l++)
        calc += 1.0f / (l+1); // para não começar do zero
    return(calc);
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned long int n=1<<30;
    unsigned long int size = n/world_size;

    double calc;
    
    // Serie 1
    if (rank != 0) {
        calc = calc_serie_1( ((rank-1)*size), rank*size );
        MPI_Send(&calc, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        calc = calc_serie_1((world_size-1)*size,n);
        double tmp;
        for(unsigned int i=1;i<world_size;i++) {
            MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            calc += tmp;
        }
        std::cout << std::endl;
        std::cout << " Serie 1 = 1 + 1/2 + 1/4 + 1/8 + 1/6 + ... " << std::endl;
        std::cout << " Tamanho do N = " <<  n << std::endl;
        std::cout << " serie 1 = " <<  std::fixed << std::setprecision(17) << calc << std::endl;
        std::cout << " serie converge para 2.0 " << std::endl << std::endl;
    }

    // Serie 2
    if (rank != 0) {
        calc = calc_serie_2( ((rank-1)*size), rank*size );
        MPI_Send(&calc, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        calc = calc_serie_2((world_size-1)*size,n);
        double tmp;
        for(unsigned int i=1;i<world_size;i++) {
            MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            calc += tmp;
        }
        std::cout << std::endl << std::endl;
        std::cout << " Serie 2 = 1 + 1/2 + 1/3 + 1/4 + 1/5 + ... " << std::endl;
        std::cout << " Tamanho do N = " <<  n << std::endl;
        std::cout << " serie 2 = " <<  std::fixed << std::setprecision(17) << calc << std::endl;
        std::cout << " serie converge para o infinito " << std::endl << std::endl;
    }

    MPI_Finalize();

    return 0;
}