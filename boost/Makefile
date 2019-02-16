CXX=g++
FLAGS=-std=c++11

all : exemplo-scoped_ptr exemplo-shared_ptr tarefa1 tarefa2

exemplo-scoped_ptr : exemplo-scoped_ptr.cpp
	$(CXX) $(FLAGS) -o $@ $^

exemplo-shared_ptr : exemplo-shared_ptr.cpp
	$(CXX) $(FLAGS) -o $@ $^

tarefa1 : tarefa1.cpp
	$(CXX) $(FLAGS) -o $@ $^

tarefa2 : tarefa2.cpp
	$(CXX) $(FLAGS) -o $@ $^

clean:
	rm -f exemplo-scoped_ptr exemplo-shared_ptr tarefa1 tarefa2
