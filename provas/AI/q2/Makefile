#Makefile para programas do curso do Supercomputação
CXX=g++
FLAGS=-std=c++11 -fopenmp -O3 -Wall -I/usr/local/include
PROGS=quicksort

all : $(PROGS)

quicksort : quicksort.cpp
	$(CXX) $(FLAGS) $^ -o $@

clean:
	rm -f $(PROGS)