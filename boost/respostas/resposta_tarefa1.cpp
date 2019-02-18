#include <boost/scoped_ptr.hpp>
#include <iostream>

int main() {
  boost::scoped_ptr<int> ptr{new int(0)};
  for(int f=0;f<1024*1024*1024;f++) {
    ptr.reset(new int(f));
  }
  std::cout << "valor final = " << *ptr << std::endl;
  ptr.reset();
}