#include <iostream>
#include <omp.h>
#include "random.h"

static long num_trials = 100000000;

int main ()
{
   long i;  long Ncirc = 0;
   double pi, x, y, test, time;
   double r = 1.0;   // radius of circle. Side of squrare is 2*r 

   time = omp_get_wtime();
   #pragma omp parallel
   {

      #pragma omp single
          std::cout << omp_get_num_threads() << " threads";

      seed(-r, r);  
      #pragma omp for reduction(+:Ncirc) private(x,y,test)
      for(i=0;i<num_trials; i++)
      {
         x = drandom(); 
         y = drandom();

         test = x*x + y*y;

         if (test <= r*r) Ncirc++;
       }
    }

    pi = 4.0 * ((double)Ncirc/(double)num_trials);

    std::cout << std::endl << num_trials << " trials, pi is " << pi;
    std::cout << " in " << omp_get_wtime()-time << " seconds\n";

}
