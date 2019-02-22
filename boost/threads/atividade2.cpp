//g++ atividade2.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt -lboost_timer-mt
// Referência: https://www.geeksforgeeks.org/bubble-sort/

#include <iostream> 
#include <boost/timer/timer.hpp>

void swap(int *xp, int *yp) { 
    int temp = *xp; 
    *xp = *yp; 
    *yp = temp; 
} 
  
void bubbleSort(int arr[], int n) { 
   int i, j; 
   for (i = 0; i < n-1; i++)       
       for (j = 0; j < n-i-1; j++)  
           if (arr[j] > arr[j+1]) 
              swap(&arr[j], &arr[j+1]); 
} 

void printArray(int arr[], int size) { 
    int i; 
    for (i=0; i < size; i++) 
        std::cout << arr[i] << std::endl;
    std::cout << std::endl;
} 

#define SIZE 40000

int main() { 
    boost::timer::cpu_timer timer;
    int arr[SIZE];
    for(int i=0;i<SIZE;i++) arr[i]=SIZE-i;
    std::cout << "Antes" << std::endl;
    printArray(arr, 10);  
    bubbleSort(arr, SIZE); 
    std::cout << "Depois" << std::endl;
    printArray(arr, 10);  
    std::cout << timer.format();
} 