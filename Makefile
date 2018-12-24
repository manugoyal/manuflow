CC=g++
CPPFLAGS += -isystem -I$(CURDIR)
CFLAGS=-I$(CURDIR) -g -O0 -Wall -std=c++11

all: matrix_test

clean:
	rm *.o matrix_test

tests_main.o: tests_main.cc
	$(CC) $(CPPFLAGS) $(CFLAGS) -o tests_main.o -c tests_main.cc

matrix_test: matrix_test.cc tests_main.o
	$(CC) $(CFLAGS) -o matrix_test matrix_test.cc tests_main.o

