// g++ threads5.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt
// Referência: https://theboostcpplibraries.com/boost.thread-management

#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <iostream>

void thread() {
  try {
    for(int i = 0; i < 5; ++i) {
      boost::this_thread::sleep_for(boost::chrono::seconds{1});
      std::cout << i << '\n';
    }
  } catch (boost::thread_interrupted&) {}
}

int main() {
  boost::thread t{thread};
  boost::this_thread::sleep_for(boost::chrono::seconds{3});
  t.interrupt();
}