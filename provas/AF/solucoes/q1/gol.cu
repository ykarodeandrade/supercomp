#include <iostream>
#include <unistd.h>
#define size 21  // Tamanho da matrix

__global__ void jogo_gpu(bool *grid) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  
  if ( i>0 && i<size-1 && j>0 && j<size-1) {
    unsigned int count = 0;
    int pos = j*size + i;
    for(int k = -1; k <= 1; k++) {
        for(int l = -size; l <= size; l+=size) {
          if(k != 0 || l != 0)
            if(grid[pos+k+l])
              ++count;
        }
    }    
    __syncthreads();
    if(count < 2 || count > 3) grid[pos+k+l] = false;
    else if(count == 3) grid[pos+k+l] = true;

  }
}

// Exibe os pontos na tela
void print(bool grid[][size]){
  std::cout << "\n\n\n\n\n";
  for(unsigned int i = 1; i < size-1; i++) {
    for(unsigned int j = 1; j < size-1; j++)
      std::cout << (grid[i][j]?"#":"_");
    std::cout << std::endl;
  }
}

// Calcula a simulacao
bool alive(bool grid[][size]){
  for(unsigned int i = 1; i < size-1; i++)
    for(unsigned int j = 1; j < size-1; j++)
      if(grid[i][j]) return true;
  return false;
}

int main(){
  bool grid[size][size] = {}; // dados iniciais
  grid[ 5][ 7] = true;
  grid[ 6][ 8] = true;
  grid[ 8][ 8] = true;
  grid[ 6][ 9] = true;
  grid[ 8][10] = true;
  grid[ 9][10] = true;
  grid[ 8][11] = true;
  grid[10][11] = true;
  grid[10][12] = true;
  
  bool *grid_d;
  print(grid);
  cudaMalloc((void **)&grid_d, size*size*sizeof(bool));
  cudaMemcpy(grid_d, grid, size*size*sizeof(bool), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(size, size);
  dim3 numBlocks(1, 1);

  while(alive(grid)) {
    jogo_gpu<<<numBlocks,threadsPerBlock>>>(grid_d);
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess ) std::cerr << "Err: " << cudaGetErrorString(err) <<std::endl;

    cudaMemcpy(grid, grid_d, size*size*sizeof(bool), cudaMemcpyDeviceToHost);  
    print(grid);
    usleep(100000);  // pausa para poder exibir no terminal
  } 
  
}
