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

Word2vec::Word2vec(string _model, string _train_method, int _iter, int _num_threads, int _layer1_size, int _window, int _negative, int _min_count, float _sample, float _alpha){
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
	cout << "alpha : " << alpha << endl;
}

int Word2vec::learn_vocab_from_trainfile(const string train_file) {
	ifstream fin(train_file.c_str(), ios::in);
	if (!fin) {
		cerr << "Can't read file " << train_file << endl;
		return -1;
	}
	unordered_map<string, long long> word_cnt;
	string word;
	while (fin >> word) {
		word_cnt[word] += 1;
	}
	fin.close();
	// remove word that cnt < min_cnt
	for (auto iter = word_cnt.begin(); iter != word_cnt.end(); iter++) {
		if (iter->second >= min_count) {
			vocab_word* vw = new vocab_word(iter->first, iter->second);
			vocab.push_back(vw);
		}
	}
	long long vocab_size = vocab.size();
	cout << "vocab size = " << vocab_size << endl;
	sort(vocab.begin(), vocab.end(), vocab_cmp);
	return 1;
}

void Word2vec::train_model_thread(const string filename, int t_id) {

}

void Word2vec::train_model(const string train_file){

}

void Word2vec::save_vector(const string output_file) {

}

