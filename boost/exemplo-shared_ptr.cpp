// Exemplo de uso de "shared_ptr"
// Referência: Boris Schäling
// Atualização: Luciano P Soares

#include <boost/shared_ptr.hpp>
#include <iostream>

int main() {
  boost::shared_ptr<char> ptr1{new char('a')}; // inicia ptr1 com 'a'
  std::cout << "ptr1 : " << *ptr1 << std::endl; // recupera o valor de ptr1
  boost::shared_ptr<char> ptr2{ptr1}; // criar ponteiro referenciando ptr1
  ptr1.reset(new char('b')); // substitui ptr1 com 'b'
  std::cout << "ptr1 : " << *ptr1.get() << std::endl; // novamente recupera o valor de ptr1
  ptr1.reset(); // reseta a referência para ptr1 (deleta)
  std::cout << "ptr2 : " << *ptr2.get() << std::endl; // recupera o valor de ptr2
  std::cout << "ptr2 : " << (static_cast<bool>(ptr2)?"cheio":"vazio") << std::endl;
  ptr2.reset(); // reseta a referência para ptr2 (deleta)
  std::cout << "ptr2 : " << (static_cast<bool>(ptr2)?"cheio":"vazio") << std::endl;
}
