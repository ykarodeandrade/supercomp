#Makefile para programas do curso do Supercomputação
CXX=g++
FLAGS=-std=c++11
PROGS=exemplo-scoped_ptr \
      exemplo-shared_ptr \
      exemplo-distribuicao_normal \
      exemplo-any \
      exemplo-salvando_aleatorios \
      exemplo-accumulator \
      exemplo-random \
      exemplo-matrizes \
      tarefa1 \
      tarefa2

all : $(PROGS)

exemplo-scoped_ptr : exemplo-scoped_ptr.cpp
	$(CXX) $(FLAGS) -o $@ $^

exemplo-shared_ptr : exemplo-shared_ptr.cpp
	$(CXX) $(FLAGS) -o $@ $^

exemplo-distribuicao_normal : exemplo-distribuicao_normal.cpp
	$(CXX) $(FLAGS) -o $@ $^

exemplo-any : exemplo-any.cpp
	$(CXX) $(FLAGS) -o $@ $^

exemplo-salvando_aleatorios : exemplo-salvando_aleatorios.cpp
	$(CXX) $(FLAGS) -o $@ $^ -lboost_serialization -lboost_program_options

exemplo-accumulator : exemplo-accumulator.cpp
	$(CXX) $(FLAGS) -o $@ $^

exemplo-random : exemplo-random.cpp
	$(CXX) $(FLAGS) -o $@ $^ -lboost_random

exemplo-matrizes : exemplo-matrizes.cpp
	$(CXX) $(FLAGS) -o $@ $^

tarefa1 : tarefa1.cpp
	$(CXX) $(FLAGS) -o $@ $^

tarefa2 : tarefa2.cpp
	$(CXX) $(FLAGS) -o $@ $^

clean:
	rm -f $(PROGS)


