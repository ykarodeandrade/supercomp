// g++ threads0.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt
#include <boost/thread.hpp>
#include <iostream>

int main() {
  std::cout << "thread id = ";
  std::cout << boost::this_thread::get_id() << std::endl;
  
  std::cout << "numero de nucleos de CPU = ";
  std::cout << boost::thread::hardware_concurrency() << std::endl;
}