CC=g++ -std=c++11
CFLAGS=-lm -pthread -Ofast -Wall

all: word2vec

word2vec: *.cpp *.h
	${CC} -o word2vec *.cpp ${CFLAGS}


clean:
	rm word2vec

