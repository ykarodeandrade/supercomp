#Makefile para programas do curso do Supercomputação
CXX=g++
FLAGS=-std=c++11 -mavx -O3 -Wall
PROGS=pi

all : $(PROGS)

pi : pi.cpp
	$(CXX) $(FLAGS) -D SIMD $^ -o $@

clean:
	rm -f $(PROGS)


