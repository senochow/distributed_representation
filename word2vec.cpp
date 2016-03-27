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
	cout << "model: " << model << "\t";
	cout << "train_method:" << train_method << "\t";
	cout << "iter: " << iter << "\t";
	cout << "number threads: " << num_threads << "\t";
	cout << "layer1_size: " << layer1_size << "\t";
	cout << "window: " << window << "\t";
	cout << "negative: " << negative << "\t";
	cout << "min_count: " << min_count << "\t";
	cout << "sample: " << sample << "\t";
	cout << "alpha: " << alpha << endl;
}

// vocab: vector<vocab_word>
int Word2vec::learn_vocab_from_trainfile(const string train_file) {
	ifstream fin(train_file.c_str(), ios::in);
    long long train_words = 0;
	if (!fin) {
		cerr << "Can't read file " << train_file << endl;
		return -1;
	}
	unordered_map<string, long long> word_cnt;
	string word;
    clock_t start, now;
    start = clock();
	while (fin >> word) {
        train_words++;
		word_cnt[word] += 1;
        if (train_words%1000000 == 0) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
	}
    now = clock();
    cout << "Loading words end !  consume time : " << static_cast<float>(now-start)/CLOCKS_PER_SEC << "s" << endl;
	fin.close();
	// remove word that cnt < min_cnt
	for (auto iter = word_cnt.begin(); iter != word_cnt.end(); iter++) {
		if (iter->second >= min_count) {
			vocab_word* vw = new vocab_word(iter->first, iter->second);
			vocab.push_back(vw);
		}
	}
	vocab_size = vocab.size();
	cout << "vocab size = " << vocab_size << endl;
	cout << "words in training file: " << train_words << endl;
	sort(vocab.begin(), vocab.end(), vocab_cmp);
	return 1;
}

void Word2vec::creat_huffman_tree() {
	long long nodes_cnt = 2 * vocab_size;
	long long i, j, min1, min2, pos1 = vocab_size-1, pos2 = vocab_size;
	vector<int> binary(nodes_cnt, 0);
	vector<long long> count(nodes_cnt);
	vector<long long> parent_node(nodes_cnt, 0);
	for (i = 0; i < vocab_size; i++) count[i] = vocab[i]->cnt;
	for (i = vocab_size; i < nodes_cnt; i++) count[i] = 1e15;
	// create huffman tree, each time select two smallest node
	for (i = 0; i < vocab_size; i++) {
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) min1 = pos1--;
			else min1 = pos2++;
		}else min1 = pos2++;
		// select second
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) min2 = pos1--;
			else min2 = pos2++;
		}else min2 = pos2++;
		count[vocab_size+i] = count[min1] + count[min2];
		parent_node[min1] = vocab_size+i;
		parent_node[min2] = vocab_size+i;
		binary[min2] = 1;
	}
	// search path and set code & path
	long long index;
	int code_len;
    int* code = new int[100];
	long long * point = new long long[100];
	for (i = 0; i < vocab_size; i++) {
		index = i;
        code_len = 0;
		while (1) {
			code[code_len] = binary[index];
			point[code_len] = index;
            code_len++;
			index = parent_node[index];
			if (index == 2*vocab_size-2) break;
		}
		vocab[i]->code_len = code_len;
		vocab[i]->point.push_back(vocab_size-2); // 0 is the root word, the last one is the index himselft
		for (j = code_len-1; j >= 0; j--) {
			vocab[i]->code.push_back(code[j]);
			vocab[i]->point.push_back(point[j]-vocab_size);
		}
	}
}

// init and creat tree: word vectors: syn0,  non-leaf node: syn1
void Word2vec::init_network() {
	syn0 = new float[vocab_size*layer1_size];
	if (train_method == "hs") {
		syn1 = new float[vocab_size*layer1_size];
		for (int i = 0; i < vocab_size; i++) {
			for (int j = 0; j < layer1_size; j++) syn1[i*layer1_size + j] = 0;
		}
	}
    // init word vectors 
	float init_bound = 1.0f/layer1_size;
	for (int i = 0; i < vocab_size; i++) {
		for (int j = 0; j < layer1_size; j++) syn0[i*layer1_size+j] = init_bound*(static_cast<float>(rand())/RAND_MAX - 0.5f);
	}
	creat_huffman_tree();
}

void Word2vec::train_model_thread(const string filename, int t_id) {

}

void Word2vec::train_model(const string train_file){
	init_network();

}

void Word2vec::save_vector(const string output_file) {

}

