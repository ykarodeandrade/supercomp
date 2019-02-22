// g++ threads8.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt
// Referência: https://theboostcpplibraries.com/boost.thread-management

#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <iostream>
using boost::this_thread::get_id;

void wait(int seconds){
  boost::this_thread::sleep_for(boost::chrono::seconds{seconds});
}

boost::timed_mutex mutex;

void thread1() {
  for(int i = 0; i < 5; ++i){
    boost::unique_lock<boost::timed_mutex> lock{mutex};
    wait(1);
    std::cout << "Thread 1 " << get_id() << ": " << i << std::endl;
    boost::timed_mutex *m = lock.release();
    m->unlock();
  }
}

void thread2() {
  for(int i = 0; i < 5; ++i) {
    boost::unique_lock<boost::timed_mutex> lock{mutex, boost::try_to_lock};
    wait(1);
    if (lock.owns_lock() || lock.try_lock_for(boost::chrono::seconds{1})) {
      std::cout << "Thread 2 " << get_id() << ": " << i << std::endl;
    }
  }
}

int main() {
  boost::thread t1{thread1}; boost::thread t2{thread2};
  t1.join(); t2.join();
}