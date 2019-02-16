// Exemplo de uso de "scoped_ptr"
// Referência: Boris Schäling
// Atualização: Luciano P Soares

#include <boost/scoped_ptr.hpp>
#include <iostream>

int main() {
  boost::scoped_ptr<char> ptr{new char('a')}; // inicia com 'a'
  std::cout << "ptr : " << *ptr.get() << std::endl; // recupera o valor
  ptr.reset(new char('b')); // substitui com 'b'
  std::cout << "ptr : " << *ptr << std::endl; // também recupera o valor
  ptr.reset(); // reseta a referência (deleta)
  std::cout << "ptr : " << (static_cast<bool>(ptr)?"cheio":"vazio") << std::endl;
}
