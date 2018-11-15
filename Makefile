CC=g++
CFLAGS=-I/home/manu/scratch/manuflow -g -O0 -Wall -std=c++11

all: test

clean:
	rm *.o test

test: test.cc matrix
	$(CC) $(CFLAGS) -o test test.cc

matrix: matrix.h matrix_impl.h
