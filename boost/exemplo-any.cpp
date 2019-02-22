#include <iostream>
#include <boost/any.hpp>
#include <vector>
using namespace std;

void callAny(boost::any x) {
	if(x.type()==typeid(double)) {
		cout << boost::any_cast<double>(x) << endl;
	}
}

int main(){
    boost::any x;
    x=string("1.1");
    x=vector<double >(3);
    x=1.1;
    if (!x.empty()) cout << x.type().name() << endl;

    double y = boost::any_cast<double>(x);
    callAny(y);
}
