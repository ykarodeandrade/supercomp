#Makefile para programas do curso do Supercomputação
CXX=g++
FLAGS=-std=c++11 -mavx -O3 -Wall
PROGS=pi pi_p

all : $(PROGS)

pi : pi.cpp
	$(CXX) $(FLAGS) -D SIMD $^ -o $@

pi_p : pi_p.cpp
	$(CXX) $(FLAGS) -D SIMD $^ -o $@

clean:
	rm -f $(PROGS)


