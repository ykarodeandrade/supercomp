// exemplo mostrando como multiplicar um vetor por uma matriz
// Referencia: https://pt.wikipedia.org/wiki/Biblioteca_Boost

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
 
using namespace boost::numeric::ublas;

// Multiplicação de uma matriz 3x3 e um vetor 3 
int main () {
    vector<double> x (3);
    x(0) = 1; x(1) = 2; x(2) = 3;
 
    matrix<double> A(3,3);
    A(0,0) = 0; A(0,1) = 1;A(0,2) = 2;
    A(1,0) = 3; A(1,1) = 4;A(1,2) = 5;
    A(2,0) = 6; A(2,1) = 7;A(2,2) = 8;

    vector<double> y = prod(A, x);
 
    std::cout << y << std::endl;
}