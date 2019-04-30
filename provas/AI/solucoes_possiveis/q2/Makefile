#Makefile para programas do curso do Supercomputação
CXX=g++-6
FLAGS=-std=c++11 -fopenmp -O3 -Wall -I/usr/local/include
PROGS=quicksort quicksort_p

all : $(PROGS)

quicksort : quicksort.cpp
	$(CXX) $(FLAGS) $^ -o $@

quicksort_p : quicksort_p.cpp
	$(CXX) $(FLAGS) $^ -o $@

clean:
	rm -f $(PROGS)