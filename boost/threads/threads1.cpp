// g++ threads1.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt
#include <iostream>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>

void thread1() {
	std::cout << "entrando na thread1" << std::endl;
	boost::this_thread::sleep_for(boost::chrono::seconds{1});
}

void thread2() {
	std::cout << "entrando na thread2" << std::endl;
	boost::this_thread::sleep_for(boost::chrono::seconds{1});
}

int main() {
  boost::thread t1{thread1};
  boost::thread t2{thread2};
  t1.join();
  t2.join();
}