/*
Este programa computará numericamente a integral de

                   4/(1 + x*x)

de 0 a 1. O valor desta integral é pi;
que é ótimo, uma vez que nos dá uma maneira fácil de verificar a resposta.

Esta versão do programa usa um algoritmo dividir e conquista e recursão.

History: Written by Tim Mattson, 10/2013
Atualizado por: Luciano Soares, 03/2019
*/

#include <omp.h>
#include <iostream>

static long num_steps = 1024*1024*1024;
#define MIN_BLK  1024*1024*256

double pi_comp(int Nstart,int Nfinish,double step) {
   int i,iblk;
   double x, sum = 0.0,sum1, sum2;
   if (Nfinish-Nstart < MIN_BLK){
      for (i=Nstart;i< Nfinish; i++){
         x = (i+0.5)*step;
         sum = sum + 4.0/(1.0+x*x); 
      }
   }
   else{
      iblk = Nfinish-Nstart;
      #pragma omp task shared(sum1)
         sum1 = pi_comp(Nstart,         Nfinish-iblk/2,step);
      #pragma omp task shared(sum2)
         sum2 = pi_comp(Nfinish-iblk/2, Nfinish,       step);
      #pragma omp taskwait
         sum = sum1 + sum2;
   }
   return sum;
}

int main () {
   int i;
   double step, pi, sum;
   double init_time, final_time;
   step = 1.0/(double) num_steps;

   init_time = omp_get_wtime();
   #pragma omp parallel
      #pragma omp single
         sum = pi_comp(0,num_steps,step);
   pi = step * sum;
   final_time = omp_get_wtime() - init_time;
   std::cout << "for " << num_steps << " steps pi = " << pi << " in " << final_time << " secs\n";
}
