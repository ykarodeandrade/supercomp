// g++ threads100.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt

#include <iostream>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>

long int fib(long int n) { 
   double x[8];
   std::cout << "endereco = " << x << '\n';
   if (n <= 1) 
      return n; 
   return fib(n-1) + fib(n-2); 
}

void thread() {
  try {
      std::cout << fib(43) << '\n';
  } catch (boost::thread_interrupted&) {}
}

int main() {
  boost::thread::attributes attrs;
  attrs.set_stack_size(8192); // minimo no meu Macbook
  std::cout << "stack size = " << attrs.get_stack_size() << '\n';
  boost::thread t{attrs, thread};
  t.join();
}