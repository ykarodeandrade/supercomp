#include <functional>
#include <iostream>

int main() {
    int c = 2;
    std::function<double(int)> by_two = [=](int n) {
        return double(n) / c; 
    };
    std::cout << by_two(5) << "\n";
    
    std::function<double(int)> by_c = [&](int n) {
        return double(n) / c; 
    };
    std::cout << by_c(7) << "\n";
    c = 3;
    std::cout << by_c(7) << "\n";
    
    return 0;
}
