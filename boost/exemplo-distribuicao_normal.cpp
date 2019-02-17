// Referência: https://www.boost.org/doc/libs/1_69_0/libs/math/doc/html/math_toolkit/stat_tut/weg/normal_example/normal_misc.html

#include <boost/math/distributions.hpp> // distributions (tem a distribuição normal)
using boost::math::normal; // typedef provides default type is double.
#include <iostream>
  using std::cout; using std::endl; using std::left; using std::showpoint; using std::noshowpoint;
#include <iomanip>
  using std::setw; using std::setprecision;
#include <limits>
  using std::numeric_limits;

int main() {

    double step = 1.0;   // in z 
    double range = 10;   // faixa = -range to +range.
    int precision = 17; // casas decimais.

    // Construindo uma distribuição normal padrão
    normal s; // (média = zero e desvio padrão = unidade)
      
    cout << "Distribuicao Normal Padrao, media = "<< s.mean()
      << ", desvio padrao = " << s.standard_deviation() << endl;

    cout << "Valores da Funcao de Distribuicao Normal" << endl;
    cout << "  z " "      pdf " << endl;
    cout.precision(5);
    for (double z = -range; z < range + step; z += step) {
      cout << left << setprecision(3) << setw(6) << z << " ";  
      cout << setprecision(precision) << setw(12) << pdf(s, z) << endl;
    }
    cout << endl;

    cout << "Integral (area sobre a curca) do -infinito ate o z" << endl;
    cout << "  z " "      cdf " << endl;
    for (double z = -range; z < range + step; z += step) {
      cout << left << setprecision(3) << setw(6) << z << " ";
      cout << setprecision(precision) << setw(12) << cdf(s, z) << endl;
    }
    
    cout.precision(5);
    cout << "95% da area esta em um z menor que: " << quantile(s, 0.95) << endl;
    cout << "95% da area esta entre um z " << quantile(s, 0.975);
    cout << " e " << -quantile(s, 0.975) << endl;

    double alpha1 = cdf(s, -1) * 2; // 0.3173105078629142
    cout << setprecision(17) << "Nivel de significancia para um z == 1 e: " << alpha1 << endl;

}