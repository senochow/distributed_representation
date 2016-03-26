/* ############################################################################
* 
* Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
* 
* ###########################################################################
* Brief: 
 
* Authors: zhouxing(@ict.ac.cn)
* Date:    2016/03/26 11:43:30
* File:    word2vec.cpp
*/
#include "word2vec.h"

Word2vec(string _model, string _train_method, int _iter, int _num_threads, int _layer1_size, int _window, int _negative, int _min_count, float _sample, float _alpha){
	model = _model;
	train_method = _train_method;
	iter = _iter;
	num_threads = _num_threads;
	layer1_size = _layer1_size;
	window = _window;
	negative = _negative;
	min_count = _min_count;
	sample = _sample;
	alpha = _alpha;
	cout << "parameters \n";
	cout << "model : " << model << endl;
	cout << "train_method :" << train_method << endl;
	cout << "iter : " << iter << endl;
	cout << "number threads : " << num_threads << endl;
	cout << "layer1_size: " << layer1_size << endl;
	cout << "window : " << window << endl;
	cout << "negative: " << negative << endl;
	cout << "min_count : " << min_count << endl;
	cout << "sample : " << sample << endl;
	cuot << "alpha : " << alpha << endl;
}

