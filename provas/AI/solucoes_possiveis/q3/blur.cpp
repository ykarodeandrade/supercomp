/*
Questão 3:
O projeto de Ray Tracing gerou imagens que apresentam um aspecto pontilhado
O seguinte código aplica um filtro de blur sobre a imagem, mas ele está sequencial.
Paralelize o código com OpenMP e faça as medidas de tempo para verificar se melhorou.
*/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include <omp.h>

#include "image.h" 

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

// Filtro de bordas
void blurFilter(int *in, int *out, int rowStart, int rowEnd, int colStart, int colEnd, int channels, int kernel) {
   int i,j,di,dj,dk;
   #pragma omp parallel for collapse(2) private(di, dj, dk) 
   for(i = rowStart; i < rowEnd; ++i) {
      for(j = colStart; j < colEnd; ++j) {
         int avg[] = {0,0,0};
         int count = 0;

         for(di = MAX(rowStart, i - kernel); di <= MIN(i + kernel, rowEnd - kernel); di++) {
            for(dj = MAX(colStart, j - kernel); dj <= MIN(j + kernel, colEnd - kernel); dj++) {
               for(dk = 0; dk < channels; dk++) {
                  avg[dk] += in[(di*(colEnd-colStart)*channels)+(dj*channels)+dk];
               }
               count++;
            }
         }
         for(dk = 0; dk < channels; dk++) {
            out[(i*(colEnd-colStart)*channels)+(j*channels)+dk] = avg[dk]/count;
         }
      }
   }
}

int main(int argc, char** argv) {

   // Estruturas que organizam as imagens PGM
   PGMData *imagemIn = (PGMData *)malloc(sizeof(PGMData));
   PGMData *imagemOut = (PGMData *)malloc(sizeof(PGMData));

   readImagePPM(argv[1],imagemIn, 3);

   createImage(imagemOut, imagemIn->row, imagemIn->col, imagemIn->max_gray, 3);

   double start_time, run_time;
   start_time = omp_get_wtime();

   // Processa os dados da imagem para a deteccao de borda
   blurFilter(imagemIn->matrix, imagemOut->matrix, 0, imagemIn->row, 0, imagemIn->col, 3, 3);

   run_time = omp_get_wtime() - start_time;
   printf("Tempo decorrido do filtro de blur: %fs\n",run_time);

   writeImagePPM(argv[2],imagemOut,3,0);

   return 0;
}
