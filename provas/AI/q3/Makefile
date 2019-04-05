#Makefile para programas do curso do Supercomputação
CXX=g++-6
FLAGS=-std=c++11 -fopenmp -O3 -Wall -I/usr/local/include
PROGS=blur
OBJS=blur.o image.o

all : $(PROGS)

blur.o : blur.cpp
	$(CXX) -c $(FLAGS) $^ -o $@

image.o : image.c
	$(CXX) -c $(FLAGS) $^ -o $@

blur : image.o blur.o
	$(CXX) $(FLAGS) $^ -o $@

clean:
	rm -f $(PROGS) $(OBJS)