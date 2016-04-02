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
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <thread>
using namespace std;



struct vocab_word {
	string word;
	long long cnt;
	int code_len;
	vector<int> code; // huffman code
	vector<int> point; // huffman path's word index
	vocab_word(string word_, long long cnt_): word(word_), cnt(cnt_) {};
};

inline bool vocab_cmp(const vocab_word* w1, const vocab_word* w2) {
	return w1->cnt > w2->cnt;
}
class Word2vec {
private:
	string model;
	string train_method;
	int iter;
	int num_threads;
	int layer1_size;
	int window;
	int negative;
	int min_count;
	float sample;
	float alpha;
private:
	vector<vocab_word *> vocab;
	unordered_map<string, long long> word2idx;
	long long vocab_size;
    long long total_words;
    long long trained_words; // processed words currently
	int max_sentence_len;
    clock_t start;
    float start_alpha;
    float min_alpha;
	float* syn0;
	float* syn1;
	// negative sampling
	float* syn1_negative;
	int* table;
	int table_size;
    int max_code_len;
    // function
	void train_model_thread(const string filename, int t_id);
	bool read_line(vector<long long>& words, int& cur_words, ifstream& fin, long long end);
	void train_cbow(vector<long long>& words, float cur_alpha);
	void train_skip_gram(vector<long long>& words, float cur_alpha);
    void init_sample_table();
	void init_network();
	void creat_huffman_tree();
public:
	Word2vec(string model, string train_method, int iter, int num_threads, int layer1_size, int window, int negative, int min_count, float sample, float alpha);
	int learn_vocab_from_trainfile(const string& train_file);	
	void train_model(const string& train_file);
	void save_vector(const string& output_file);

};	

#endif
