#ifndef WORD2VEC_H
#define WORD2VEC_H
/* ############################################################################
* 
* Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
* 
* ###########################################################################
* Brief: 
 
* Authors: zhouxing(@ict.ac.cn)
* Date:    2016/03/26 11:45:17
* File:    word2vec.h
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

class Word2vec {
public:
	string model;
	string train_method;
	int iter;
	int num_threads;
	int layer1_size;
	int window;
	int negtive;
	int min_count;
	float sample;
	float alpha;
public:
	Word2vec(string model, string train_method, int iter, int num_threads, int layer1_size, int window, int negative, int min_count, float sample, float alpha);
};	

#endif