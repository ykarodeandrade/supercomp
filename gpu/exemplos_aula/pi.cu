#include <iostream>

__global__ void pi(double step, int N, double *value) {
    __shared__ double sum;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) {   // Importante checar valor do i pois pode acessar fora do tamanho do vetor
        double x = (i+0.5)*step;               
        *value += 4.0/(1.0+x*x);
    }
}

static long num_steps = 100000000;

int main() {
    
   double step;

   double *h_a;
   double *d_a;
   int    blocksize;

   step = 1.0 / (double)num_steps;

   // Aloca vetores na memoria da CPU
   h_a = (double *)calloc(1,sizeof(double));

   // Aloca vetores na memoria da GPU
    cudaMalloc((void **)&d_a,sizeof(double));

    // Copia valores da CPU para a GPU
    cudaMemcpy(d_a, h_a, sizeof(double), cudaMemcpyHostToDevice);
   
    // Realiza calculo na GPU
    blocksize = 256;
    pi<<<((num_steps-1)/256 + 1),blocksize>>>(step, num_steps, d_a);

    // Retorna valores da memoria da GPU para a CPU
    cudaMemcpy(h_c, d_c, sizeof(double), cudaMemcpyDeviceToHost);

    

    // Libera memoria da GPU
    cudaFree(d_a);
   
    // Exibe um resultado para checar se valores conferem
    std::cout << "pi = " << *h_a << std::endl;
   
   // Libera memoria da CPU
   free(h_a);

   return 0;
}
