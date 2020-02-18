#ifndef FUNC_H
#define FUNC_H

#include <vector>
#include <memory>

std::shared_ptr<double[]> cria_vetor(int n);
void processa(std::shared_ptr<double[]>ptr, int n);

#endif
