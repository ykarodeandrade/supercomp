#include <iostream>
#include <boost/any.hpp>
#include <vector>

void callAny(boost::any x) {
    std::cout << boost::any_cast<double>(x) << '\n';
}

int main(){
    boost::any x;
    x=std::string("1.1");
    x=std::vector<double >(3);
    x=1.1;
    if (!x.empty()) std::cout << x.type().name() << '\n';

    double y = boost::any_cast<double>(x);
    callAny(y);
}