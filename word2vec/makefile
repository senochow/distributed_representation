CC=g++
CFLAGS=-std=c++11 -lm -pthread -Ofast -Wall

word2vec: *.cpp *.h
	${CC} -o word2vec *.cpp ${CFLAGS}

clean:
	rm word2vec

