#include <iostream>
#include <mpi.h>


double pi(double x) {
    return (4/(1+x*x));
}

double integral(double (*f)(double), double a, double b, double div=100000) {
    double step = (b-a)/div;
    double y1,y2;
    double parcial=0;
    std::cout << "[" << a << "," << b << "]\n";
    for(double i=a;i<b-step;i+=step) {
        y1=(*f)(i);
        y2=(*f)(i+step);
        parcial+=((y1+y2)*step)/2;
    }
    return parcial;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size; MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n=world_size-1;

    double faixa[] = {0,1};

    double *parc = new double[n];
    double soma=0;
    if(world_rank==0) {
     for(int i=0;i<n;i++) {
      MPI_Recv(&parc[i], 1, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      soma+=parc[i];
     }
    std::cout << "PI = " << soma << std::endl;
    } else {
      double step = (faixa[1]-faixa[0])/n;
      double tmp = integral(pi, (world_rank*step)-step, world_rank*step);
      MPI_Send(&tmp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    delete[] parc;
    MPI_Finalize();
    return 0;
}