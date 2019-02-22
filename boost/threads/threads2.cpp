// g++ threads2.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt
#include <iostream>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>

void thread(int n) {
	std::cout << "entrando na thread" << n << std::endl;
	boost::this_thread::sleep_for(boost::chrono::seconds{1});
}

int main() {
  boost::thread t1{thread,1};
  boost::thread t2{thread,2};
  t1.join();
  t2.join();
}