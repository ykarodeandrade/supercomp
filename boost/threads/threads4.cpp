// g++ threads4.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt
#include <iostream>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/thread/scoped_thread.hpp>

void thread(int n) {
	std::cout << "entrando na thread" << n << std::endl;
	boost::this_thread::sleep_for(boost::chrono::seconds{1});
	std::cout << "saindo da thread" << n << std::endl;
}

int main() {
  boost::scoped_thread<> t1{boost::thread{thread,1}};
  boost::scoped_thread<> t2{boost::thread{thread,2}};
}