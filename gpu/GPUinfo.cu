// Exemplo para o curso de Super Computação
// Criado por: Luciano P. Soares (10 de Abril de 2018)
// Atualizado em 22 de Abril de 2019

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

/* Informacoes da GPU */
int main() {

   int dev_count;
   cudaGetDeviceCount(&dev_count);
   std::cout << "Numero de devices (GPU) = " << dev_count << std::endl;

   cudaDeviceProp dev_prop;
   for (int i = 0; i < dev_count; i++) {
      std::cout << "\n\tDevice (" << i << ")\n";
      
      cudaGetDeviceProperties(&dev_prop, i);

      std::cout << "\t\tDimensão máxima de blocos em x = " <<  dev_prop.maxGridSize[0] << ", y = " <<  dev_prop.maxGridSize[1] << ", z = " <<  dev_prop.maxGridSize[2] << std::endl;
      std::cout << "\t\tNúmero máximo de Threads por Bloco = " << dev_prop.maxThreadsPerBlock << std::endl;
      std::cout << "\t\tDimensão máxima em x = " << dev_prop.maxThreadsDim[0] << ", y = " << dev_prop.maxThreadsDim[1] << ", z = " << dev_prop.maxThreadsDim[0] << std::endl;
      std::cout << "\t\tNúmero máximo de Streaming Multiprocessors (SMs) = " << dev_prop.multiProcessorCount << std::endl;
      std::cout << "\t\tFrequência de Clock = " << dev_prop.clockRate << std::endl;
      std::cout << "\t\tTamanho do Warp = " << dev_prop.warpSize << std::endl;
   
   }

   return 0;
}

