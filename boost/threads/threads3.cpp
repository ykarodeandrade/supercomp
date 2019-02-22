// g++ threads3.cpp -std=c++14 -lboost_thread-mt -lboost_chrono-mt -lboost_system-mt
#include <iostream>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>

class classThread {
    int id;
    public:
    classThread(int i) : id(i) { }  // Construtor
    void operator()() {
      std::cout << "thread dentro da classe" << id << std::endl;
      boost::this_thread::sleep_for(boost::chrono::seconds{1});
    };
};

int main() {
	classThread x(1);
  boost::thread t(x);
	t.join();
}