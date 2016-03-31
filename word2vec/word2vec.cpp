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
    total_words = 0;
	ifstream fin(train_file, ios::in);
	if (!fin) {
		cerr << "Can't read file " << train_file << endl;
		return -1;
	}
	unordered_map<string, long long> word_cnt;
	string word;
    clock_t now;
    start = clock();
	while (fin >> word) {
        total_words++;
		word_cnt[word] += 1;
        if (total_words%1000000 == 0) {
            printf("%lldK%c", total_words / 1000, 13);
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
	cout << "words in training file: " << total_words << endl;
	sort(vocab.begin(), vocab.end(), vocab_cmp);
	for (int i = 0; i < vocab_size; i++) {
		word2idx[vocab[i]->word] = i;
	}
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
// init negative sample table
void Word2vec::init_sample_table() {
	double total_words_pow = 0.0, cur_pow = 0.0, power = 0.75;
	table = new int[table_size];
	long long i = 0;
	for (i = 0; i < vocab_size; i++) total_words_pow += pow(vocab[i]->cnt, power);
	cur_pow = pow(vocab[0]->cnt, power)/total_words_pow;
	i = 0;
	for (int j = 0; j < table_size; j++) {
		table[j] = i;
		if (static_cast<double>(j)/table_size > cur_pow) {
			i++;
			cur_pow += pow(vocab[i]->cnt, power)/total_words_pow;
		}
		if (i >= vocab_size) i = vocab_size-1;
	}

}
// init and creat tree: word vectors: syn0,  non-leaf node: syn1
void Word2vec::init_network() {
    // init global variable
	max_sentence_len = 1000;
    trained_words = 0;
    long long i, j;
    table_size = 1e8;
    start_alpha = alpha;
    min_alpha = start_alpha*0.0001;
    // init network paramters
	syn0 = new float[vocab_size*layer1_size];
	if (train_method == "hs") {
		syn1 = new float[vocab_size*layer1_size];
		for (i = 0; i < vocab_size; i++) {
			for (j = 0; j < layer1_size; j++) syn1[i*layer1_size + j] = 0;
		}
	}
	// if using negative sampling , init syn1_negative
	if (negative > 0) {
		syn1_negative = new float[vocab_size*layer1_size];
		for (i = 0; i < vocab_size; i++) {
			for (j = 0; j < layer1_size; j++) syn1_negative[i*layer1_size+j] = 0;
		}
		init_sample_table();
	}
    // init word vectors 
	float init_bound = 1.0f/layer1_size;
	for (i = 0; i < vocab_size; i++) {
		for (j = 0; j < layer1_size; j++) syn0[i*layer1_size+j] = init_bound*(static_cast<float>(rand())/RAND_MAX - 0.5f);
	}
	creat_huffman_tree();
}
//read line 
bool Word2vec::read_line(vector<int>& words, ifstream& fin, long long end) {
	if (fin.eof() || fin.tellg() >= end) return false;
	string word;
	char c;
	while (static_cast<int>(words.size()) < max_sentence_len) {
		c = fin.get();
		if (fin.eof()) return true;
		if (c == ' ' || c == '\t' || c == '\n') {
			if (!word.empty()) {
				auto iter = word2idx.find(word);
				if (iter != word2idx.end()) words.push_back(iter->second);
				word.clear();
			}
			if (c == '\n') return true;
		}else word.push_back(c);
	}
	return true;

}
void Word2vec::train_cbow(vector<int>& words, float cur_alpha) {
	float* neu1 = new float[layer1_size]; // record context mean
	float* neu1e = new float[layer1_size]; // record backprob error from output->hidden
	int sent_len = words.size();
	int sample_word = 0, label = 0;
	if (sent_len <= 1) return;
	int l, q; // l for layer index
	float f, grad;
	for (int cur_word = 0; cur_word < sent_len; cur_word++) {
		// init context vector
		for (l = 0; l < layer1_size; l++) {
			neu1[l] = 0;
			neu1e[l] = 0;
		}
		// set context = (window - b)
		int b = rand()%window;// context : [1..window]
		int c_beg = max(0, cur_word-window+b);
		int c_end = min(sent_len-1, cur_word+window-b);
		int c_cnt = c_end-c_beg;
		for (int i = c_beg; i <= c_end; i++) {
			if (i == cur_word) continue;
			for (l = 0; l < layer1_size; l++) neu1[l] += syn0[words[i]*layer1_size + l];
		}
		// context mean
		for (l = 0; l < layer1_size; l++) neu1[l] /= c_cnt;
		// training hirichical softmax
		if (train_method == "hs") {
			// iter for every code
			for (int d = 0; d < vocab[cur_word]->code_len; d++) {
				f = 0;
				q = (vocab[cur_word]->point[d])*layer1_size;
				// hidden->output
				for (l = 0; l < layer1_size; l++) f += syn1[q+l]*neu1[l];
				f = 1.0/(1.0+exp(-f));
				grad = (1-vocab[cur_word]->code[d]-f)*cur_alpha;
				// propogate error output->hidden
				for (l = 0; l < layer1_size; l++) neu1e[l] += grad*syn1[l+q];
				// Learn weights hidden -> output
				for (l = 0; l < layer1_size; l++) syn1[l+q] += grad*neu1[l];
			}
		}
		// negative sampling method..
		if (negative > 0) {
			// word + k negative words
			long long l2 = 0;
			for (int k = 0; k < negative+1; k++) {
				if (k == 0) {
					sample_word = cur_word;
					label = 1;
				}else {
					sample_word = table[rand()%table_size];
					if (sample_word == cur_word) continue;
					label = 0;
				}
				l2 = sample_word*layer1_size; //position 
				f = 0;
				for (l = 0; l < layer1_size; l++) f += neu1[l] * syn1_negative[l2+l];
				f = 1.0/(1.0+exp(-f));
				grad = (label-f)*cur_alpha;
				//output->hidden
				for (l = 0; l < layer1_size; l++) neu1[l] += grad*syn1_negative[l2+l];
				// learn weights from hidden->out
				for (l = 0; l < layer1_size; l++) syn1_negative[l2+l] += grad*neu1[l];
			}
		}
		// update word vectors of word's context words , hidden-> input
		for (int i = c_beg; i <= c_end; i++) {
			if (i == cur_word) continue;
			for (l = 0; l < layer1_size; l++) syn0[words[i]*layer1_size + l] += neu1e[l];
		}
	}
}

void Word2vec::train_skip_gram(vector<int>& words, float cur_alpha) {

}
void Word2vec::train_model_thread(const string filename, int t_id) {
	ifstream fin(filename, ios::in);
    fin.seekg(0, ios::end);
	long long file_size = fin.tellg();
	// 设置当前线程文件的读取范围
	long long fbeg = file_size/num_threads*t_id, fend;
    if (t_id == num_threads-1) fend = file_size;
    else fend = file_size/num_threads*(t_id+1);
    // process file
    fin.seekg(fbeg, ios::beg);
    vector<int> words;
    clock_t now;
    long long word_cnt = 0, i = 0;
    // read each line 
    while (read_line(words, fin, fend)) {
        i++;
        word_cnt += words.size();
        if (i % 1000 == 0) {
            trained_words += word_cnt;
            now = clock();
            printf("%cAlpha: %f Progress: %.2f%% Words/thread/sec: %.2fk ", 13, alpha, 
                    static_cast<float>(trained_words)/(iter*total_words+1)*100, 
                    static_cast<float>(trained_words)/(static_cast<float>(now-start+1)/CLOCKS_PER_SEC*1000));
            fflush(stdout);
            alpha = start_alpha*(1- static_cast<float>(trained_words)/(iter*total_words+1));
            if (alpha < min_alpha) alpha = min_alpha;
            word_cnt = 0;
        }
    	if (model == "cbow") {
    		train_cbow(words, alpha);
    	}else if (model == "sg") {
    		train_skip_gram(words, alpha);
    	}
    	words.clear();
    }
    fin.close();
}

void Word2vec::train_model(const string train_file){
	ifstream fin(train_file, ios::in);
	if (!fin) {
		cerr << "Can't open file " << train_file << endl;
		return;
	}
	fin.close();
    init_network();
    start = clock();
	for (int i = 0; i < iter; i++) {
		cout << "iter " << i << endl;
		vector<thread> threads;
		for (int j = 0; j < num_threads; j++) {
			threads.push_back(thread(&Word2vec::train_model_thread, this, train_file, j));
		}
		for (int j = 0; j < num_threads; j++) threads[j].join();
		cout << endl;
	}
    clock_t now = clock();
    cout << "Total training time : " << static_cast<float>(now-start)/CLOCKS_PER_SEC << " s"<< endl;
}

void Word2vec::save_vector(const string output_file) {
	ofstream fout(output_file, ios::out);
	fout << vocab_size << " " << layer1_size << endl;
	for (long long i = 0; i < vocab_size; i++) {
		fout << vocab[i]->word;
		for (int j = 0; j < layer1_size; j++) {
			fout << " " << syn0[i*layer1_size + j];
		}
		fout << endl;
	}
	fout.close();
}
