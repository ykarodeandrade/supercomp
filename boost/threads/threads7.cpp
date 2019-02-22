// g++ threads7.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt
// Referência: https://theboostcpplibraries.com/boost.thread-management

#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <iostream>

boost::mutex mutex;

void thread() {
  using boost::this_thread::get_id;
  for(int i = 0; i < 5; ++i) {
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::lock_guard<boost::mutex> lock{mutex};
    std::cout << "Thread " << get_id() << ": " << i << std::endl;
  }
}

int main() {
  boost::thread t1{thread};
  boost::thread t2{thread};
  t1.join();
  t2.join();
}