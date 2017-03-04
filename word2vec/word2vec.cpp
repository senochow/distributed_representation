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
typedef unsigned int uint32_t;
float rsqrt(float number){
    uint32_t i;
    float x2, y;
    x2 = number * 0.5F;
    y  = number;
    i  = *(uint32_t *) &y;
    i  = 0x5f3759df - ( i >> 1 );
    y  = *(float *) &i;
    y  = y * ( 1.5F - ( x2 * y * y ) );
    return y;
}
Word2vec::Word2vec(string _model, string _train_method, int _iter, int _num_threads, int _layer1_size, int _window, int _negative, int _min_count, float _sample, float _alpha, int _adagrad){
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
    adagrad = _adagrad;
    cout << "parameters \n";
    cout << "'model': " << model << ", ";
    cout << "'train_method': " << train_method << ", ";
    cout << "'iter': " << iter << ", ";
    cout << "'number threads': " << num_threads << ", ";
    cout << "'layer1_size': " << layer1_size << ", ";
    cout << "'window': " << window << ", ";
    cout << "'negative': " << negative << ", ";
    cout << "'min_count': " << min_count << ", ";
    cout << "'sample': " << sample << ", ";
    cout << "'alpha': " << alpha << endl;
    cout << "'Adagrad': " << adagrad << endl;
}

// vocab: vector<vocab_word>
int Word2vec::learn_vocab_from_trainfile(const string& train_file) {
    total_words = 0;
    ifstream fin(train_file, ios::in);
    if (!fin) {
        cerr << "Can't read file " << train_file << endl;
        return -1;
    }
    cout <<"Loading training file : " << train_file << endl;
    unordered_map<string, long long> word_cnt;
    string word;
    struct timeb load_beg, load_end;
    ftime(&load_beg);
    while (fin >> word) {
        total_words++;
        word_cnt[word] += 1;
        if (total_words%1000000 == 0) {
            printf("%lldK%c", total_words / 1000, 13);
            fflush(stdout);
        }
    }
    ftime(&load_end);
    int load_cost_sec = load_end.time - load_beg.time;
    cout << "Loading words: "<< total_words << " end !  consume time : " << load_cost_sec/60 << " m " << load_cost_sec%60 << " s "<< endl;
    fin.close();
    // remove word that cnt < min_cnt
    // remember to reset total words cnt
    total_words = 0;
    for (auto iter = word_cnt.begin(); iter != word_cnt.end(); iter++) {
        if (iter->second >= min_count) {
            vocab_word* vw = new vocab_word(iter->first, iter->second);
            vocab.push_back(vw);
            total_words += iter->second;
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
int Word2vec::load_simwords(const string& sim_file) {
    ifstream fin(sim_file, ios::in);
    string line;
    while (getline(fin, line)) {
        stringstream ss(line);
        string word;
        long long main_wordidx;
        vector<long long> words;
        int i = 0;
        while (getline(ss, word, ' ') {
            wordindex = word2idx[word];
            if (i == 0) {
                main_wordidx = wordindex;
            } else {
                words.push_back(wordindex);
            }   
            i++;
        }
        sim_words[main_wordidx] = words;
    }
    fin.close();
    return 1;
}
// tree length is log(V)
void Word2vec::creat_huffman_tree() {
    long long nodes_cnt = 2 * vocab_size;
    long long i, j, min1, min2, pos1 = vocab_size-1, pos2 = vocab_size;
    vector<int> binary(nodes_cnt, 0);
    vector<long long> count(nodes_cnt);
    vector<long long> parent_node(nodes_cnt, 0);
    for (i = 0; i < vocab_size; i++) count[i] = vocab[i]->cnt;
    for (i = vocab_size; i < nodes_cnt; i++) count[i] = 1e15;
    // create huffman tree, each time select two smallest node
    // create n-1 non-leaf nodes 
    for (i = 0; i < vocab_size-1; i++) {
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
    vector<int> code(max_code_len);
    vector<long long> point(max_code_len);
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
    max_code_len = 40;
    table_size = 1e8;
    start_alpha = alpha;
    min_alpha = start_alpha*0.0001;
    // init network paramters
    syn0 = new float[vocab_size*layer1_size];
    if (adagrad) syn0_gdsq = new float[vocab_size*layer1_size];
    if (train_method == "hs") {
        syn1 = new float[vocab_size*layer1_size];
        if (adagrad) syn1_gdsq = new float[vocab_size*layer1_size];
        for (i = 0; i < vocab_size; i++) {
            for (j = 0; j < layer1_size; j++) {
                syn1[i*layer1_size + j] = 0;
                if (adagrad) syn1_gdsq[i*layer1_size + j] = 1e-8;
            }
        }
    }
    // if using negative sampling , init syn1_negative
    if (negative > 0) {
        syn1_negative = new float[vocab_size*layer1_size];
        if (adagrad) syn1_neg_gdsq = new float[vocab_size*layer1_size];
        for (i = 0; i < vocab_size; i++) {
            for (j = 0; j < layer1_size; j++) {
                syn1_negative[i*layer1_size+j] = 0;
                if (adagrad) syn1_neg_gdsq[i*layer1_size+j] = 1e-8;
            }
        }
        init_sample_table();
    }
    // init word vectors 
    float init_bound = 1.0f/layer1_size;
    for (i = 0; i < vocab_size; i++) {
        for (j = 0; j < layer1_size; j++) {
            syn0[i*layer1_size+j] = init_bound*(static_cast<float>(rand())/RAND_MAX - 0.5f);
            if (adagrad) syn0_gdsq[i*layer1_size+j] = 1e-8;
        }
    }
    creat_huffman_tree();
}
//read line 
bool Word2vec::read_line(vector<long long>& words, int& cur_words, ifstream& fin, long long end) {
    if (fin.eof() || fin.tellg() >= end) return false;
    string word;
    char c;
    double random = 0;
    while (static_cast<int>(words.size()) < max_sentence_len) {
        c = fin.get();
        if (fin.eof()) return true;
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!word.empty()) {
                auto iter = word2idx.find(word);
                // some infrequent words will be remove before
                //cur_words++;
                if (iter != word2idx.end()) {
                    cur_words++;
                    if (sample > 0) {
                        //  sub sampling with probability: p = 1-sqrt(sample/freq)-sample/freq
                        float p = sample*total_words/vocab[iter->second]->cnt;
                        p = sqrt(p) + p;
                        random = static_cast<double>(rand())/RAND_MAX;
                        if (p < random) {
                            word.clear();
                            continue;
                        }
                    }
                    words.push_back(iter->second);
                }
                word.clear();
            }
            if (c == '\n') return true;
        }else word.push_back(c);
    }
    return true;

}
// context predict center words, p(w_i | c_i);
// hidden layer : average of all context words of cur_word
// hs: grad = (1-code-f)*alpha
// neg: grad = (label-f)*alpha
// neu1e save error for update input context words excluded cur_word iteself
void Word2vec::train_cbow(vector<long long>& words, float cur_alpha) {
    vector<float> neu1(layer1_size, 0);
    vector<float> neu1e(layer1_size, 0); // save hidden neura errors
    int sent_len = words.size();
    if (sent_len <= 1) return;
    int l, q, label; // l for layer index
    float f, grad, g_t;
    long long cur_word, sample_word;
    // for each word index
    for (int index = 0; index < sent_len; index++) {
        cur_word = words[index];
        // init context vector
        for (l = 0; l < layer1_size; l++) {
            neu1[l] = 0;
            neu1e[l] = 0;
        }
        // set context = (window - b)
        int b = rand()%window;// context : [1..window]
        int c_beg = max(0, index-window+b);
        int c_end = min(sent_len-1, index+window-b);
        int c_cnt = c_end-c_beg;
        for (int i = c_beg; i <= c_end; i++) {
            if (i == index) continue; // pass cur_word index
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
                if (adagrad) {
                    grad = 1-vocab[cur_word]->code[d]-f;
                    // propogate error output->hidden
                    for (l = 0; l < layer1_size; l++) neu1e[l] += grad*syn1[l+q];
                    // Learn weights hidden -> output
                    for (l = 0; l < layer1_size; l++) {
                        g_t = grad*neu1[l];
                        syn1_gdsq[l+q] += g_t*g_t;
                        syn1[l+q] += cur_alpha*g_t*rsqrt(syn1_gdsq[l+q]);
                    }

                }else {
                    grad = (1-vocab[cur_word]->code[d]-f)*cur_alpha;
                    // propogate error output->hidden
                    for (l = 0; l < layer1_size; l++) neu1e[l] += grad*syn1[l+q];
                    // Learn weights hidden -> output
                    for (l = 0; l < layer1_size; l++) syn1[l+q] += grad*neu1[l];
                }
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
                if (adagrad) {
                    grad = label-f;
                    //output->hidden
                    for (l = 0; l < layer1_size; l++) neu1e[l] += grad*syn1_negative[l2+l];
                    // learn weights from hidden->out
                    for (l = 0; l < layer1_size; l++) {
                        g_t = grad*neu1[l];
                        syn1_neg_gdsq[l2 + l] += g_t*g_t;
                        syn1_negative[l2 + l] += cur_alpha*g_t*rsqrt(syn1_neg_gdsq[l2 + l]);
                    }
                }else {
                    grad = (label-f)*cur_alpha; // grad fro current instance
                    //output->hidden
                    for (l = 0; l < layer1_size; l++) neu1e[l] += grad*syn1_negative[l2+l];
                    // learn weights from hidden->out
                    for (l = 0; l < layer1_size; l++) syn1_negative[l2+l] += grad*neu1[l];
                }
            }
        }
        // update word vectors of word's context words , hidden-> input
        long long p = 0;
        for (int i = c_beg; i <= c_end; i++) {
            if (i == index) continue; // pass cur_word index 
            if (adagrad) {
                for (l = 0; l < layer1_size; l++) {
                    g_t = neu1e[l];
                    p = words[i]*layer1_size + l;
                    syn0_gdsq[p] += g_t*g_t;
                    syn0[p] += cur_alpha*g_t*rsqrt(syn0_gdsq[p]);
                }
            }else {
                for (l = 0; l < layer1_size; l++) syn0[words[i]*layer1_size + l] += neu1e[l];
            }
        }
    }
}
// skip-gram model: center word i to predict context word j where (-c <= j <= c && j != 0) ; 
// equivalent : sum_words:sum_context:p(w_j|w_i) = sum_words:sum_context:p(w_i | w_j)
// huffman: f = sigmoid(q_k * w_j), grad = (1-code[k]-f)*alpha
// bp errors from ouput to hidden: neu1e += grad*q_k
// update q_k : q_k += grad*neu1   where neu1 = w_j 
// Update input word: when using the equivalent expression, we update each context word of cur word. while cbow update all context word of cur.
void Word2vec::train_skip_gram(vector<long long>& words, float cur_alpha) {
    vector<float> neu1e(layer1_size, 0);
    int sent_len = words.size();
    long long cur_word, sample_word;
    int l = 0, d = 0, label = 0;
    float f, grad, g_t;
    long long l1, l2;
    for (int index = 0; index < sent_len; index++) {
        cur_word = words[index];
        // get context of cur_words
        int b = rand()%window;// b~[0..window-1] context: window-b ~ [1..window]
        int c_beg = max(0, index-window+b);
        int c_end = min(sent_len-1, index+window-b);
        for (int c_index = c_beg; c_index <= c_end; c_index++) {
            if (c_index == index) continue;
            for (l = 0; l < layer1_size; l++) neu1e[l] = 0;
            sample_word = words[c_index];
            l1 = sample_word*layer1_size;
            // using hs
            if (train_method == "hs") {
                for (d = 0; d < vocab[cur_word]->code_len; d++) {
                    f = 0;
                    l2 = vocab[cur_word]->point[d] * layer1_size;
                    for (l = 0; l <layer1_size; l++) f += syn0[l1 + l] * syn1[l2 + l];
                    f = 1.0/(1+exp(-f));
                    if (adagrad) {
                        grad = 1-vocab[cur_word]->code[d]-f;
                        // propogate error output->hidden
                        for (l = 0; l < layer1_size; l++) neu1e[l] += grad*syn1[l2 + l];
                        // Learn weights hidden -> output
                        for (l = 0; l < layer1_size; l++) {
                            g_t = grad*syn0[l1 + l];
                            syn1_gdsq[l2 + l] += g_t*g_t;
                            syn1[l2 + l] += cur_alpha*g_t*rsqrt(syn1_gdsq[l2 + l]);
                        }
                    }else {
                        grad = (1-vocab[cur_word]->code[d]-f)*cur_alpha;
                        // back error from output -> hidden;
                        for (l = 0; l < layer1_size; l++) neu1e[l] += grad*syn1[l2 + l];
                        // update non-leaf weights
                        for (l = 0; l < layer1_size; l++) syn1[l2 + l] += grad*syn0[l1 + l];
                    }
                }
            }
            // using negative sampling
            if (negative > 0) {
                for (int k = 0; k < negative+1; k++) {
                    if (k == 0) {
                        sample_word = cur_word;
                        label = 1;
                    }else {
                        sample_word = table[rand()%table_size];
                        if (sample_word == cur_word) continue;
                        label = 0;
                    }
                    l2 = sample_word*layer1_size;
                    f = 0;
                    for (l = 0; l < layer1_size; l++) f += syn0[l1 + l] * syn1_negative[l2 + l];
                    f = 1.0/(1+exp(-f));
                    if (adagrad) {
                        grad = (label - f);
                        // bp : ouput->hidden
                        for (l = 0; l < layer1_size; l++) neu1e[l] += grad * syn1_negative[l2 + l];
                        // update syn1_neg 
                        for (l = 0; l < layer1_size; l++) {
                            g_t = grad*syn0[l1 +l];
                            syn1_neg_gdsq[l2+l] += g_t*g_t;
                            syn1_negative[l2+l] += cur_alpha*g_t*rsqrt(syn1_neg_gdsq[l2+l]);
                        }
                    }else {
                        grad = (label - f)*cur_alpha;
                        // bp : ouput->hidden
                        for (l = 0; l < layer1_size; l++) neu1e[l] += grad * syn1_negative[l2 + l];
                        // update syn1_neg 
                        for (l = 0; l < layer1_size; l++) syn1_negative[l2+l] += grad*syn0[l1 + l];
                    }
                }
            }
            // update cur context word which is the input word.
            if (adagrad) {
                for (l = 0; l < layer1_size; l++) {
                    float g_t = neu1e[l];
                    syn0_gdsq[l1 + l] += g_t*g_t;
                    syn0[l1 + l] += cur_alpha*g_t*rsqrt(syn0_gdsq[l1+l]);
                }
            }else {
                for (l = 0; l < layer1_size; l++) syn0[l1 + l] += neu1e[l]; 
            }
        }

    }
}

// refer: Specializing Word Embeddings for Similarity or Relatedness
void Word2vec::train_skip_gram_with_specializing(vector<long long>& words, float cur_alpha) {
    vector<float> neu1e(layer1_size, 0);
    int sent_len = words.size();
    long long cur_word, sample_word;
    int l = 0, d = 0, label = 0;
    float f, grad, g_t;
    long long l1, l2;
    for (int index = 0; index < sent_len; index++) {
        cur_word = words[index];
        // get context of cur_words
        int b = rand()%window;// b~[0..window-1] context: window-b ~ [1..window]
        int c_beg = max(0, index-window+b);
        int c_end = min(sent_len-1, index+window-b);
        for (int c_index = c_beg; c_index <= c_end; c_index++) {
            if (c_index == index) continue;
            for (l = 0; l < layer1_size; l++) neu1e[l] = 0;
            sample_word = words[c_index];
            l1 = sample_word*layer1_size;
            // using negative sampling
            if (negative > 0) {
                for (int k = 0; k < negative+1; k++) {
                    if (k == 0) {
                        sample_word = cur_word;
                        label = 1;
                    }else {
                        sample_word = table[rand()%table_size];
                        if (sample_word == cur_word) continue;
                        label = 0;
                    }
                    l2 = sample_word*layer1_size;
                    f = 0;
                    for (l = 0; l < layer1_size; l++) f += syn0[l1 + l] * syn1_negative[l2 + l];
                    f = 1.0/(1+exp(-f));
                    if (adagrad) {
                        grad = (label - f);
                        // bp : ouput->hidden
                        for (l = 0; l < layer1_size; l++) neu1e[l] += grad * syn1_negative[l2 + l];
                        // update syn1_neg 
                        for (l = 0; l < layer1_size; l++) {
                            g_t = grad*syn0[l1 +l];
                            syn1_neg_gdsq[l2+l] += g_t*g_t;
                            syn1_negative[l2+l] += cur_alpha*g_t*rsqrt(syn1_neg_gdsq[l2+l]);
                        }
                    }else {
                        grad = (label - f)*cur_alpha;
                        // bp : ouput->hidden
                        for (l = 0; l < layer1_size; l++) neu1e[l] += grad * syn1_negative[l2 + l];
                        // update syn1_neg 
                        for (l = 0; l < layer1_size; l++) syn1_negative[l2+l] += grad*syn0[l1 + l];
                    }
                }
            }
            // update cur context word which is the input word.
            if (adagrad) {
                for (l = 0; l < layer1_size; l++) {
                    float g_t = neu1e[l];
                    syn0_gdsq[l1 + l] += g_t*g_t;
                    syn0[l1 + l] += cur_alpha*g_t*rsqrt(syn0_gdsq[l1+l]);
                }
            }else {
                for (l = 0; l < layer1_size; l++) syn0[l1 + l] += neu1e[l]; 
            }
        }
        // add similar words as context words
        if (sim_words.find(cur_word) != sim_words.end()) {
            for (int a = 0; a < sim_words[cur_word].size(); a++) {
                for (l = 0; l < layer1_size; l++) neu1e[l] = 0;
                    sample_word = sim_words[cur_word][a];
                    l1 = sample_word*layer1_size;
                    // using negative sampling
                    if (negative > 0) {
                        for (int k = 0; k < negative+1; k++) {
                            if (k == 0) {
                                sample_word = cur_word;
                                label = 1;
                            }else {
                                sample_word = table[rand()%table_size];
                                if (sample_word == cur_word) continue;
                                label = 0;
                            }
                            l2 = sample_word*layer1_size;
                            f = 0;
                            for (l = 0; l < layer1_size; l++) f += syn0[l1 + l] * syn1_negative[l2 + l];
                            f = 1.0/(1+exp(-f));
                            if (adagrad) {
                                grad = (label - f);
                                // bp : ouput->hidden
                                for (l = 0; l < layer1_size; l++) neu1e[l] += grad * syn1_negative[l2 + l];
                                // update syn1_neg 
                                for (l = 0; l < layer1_size; l++) {
                                    g_t = grad*syn0[l1 +l];
                                    syn1_neg_gdsq[l2+l] += g_t*g_t;
                                    syn1_negative[l2+l] += cur_alpha*g_t*rsqrt(syn1_neg_gdsq[l2+l]);
                                }
                            }else {
                                grad = (label - f)*cur_alpha;
                                // bp : ouput->hidden
                                for (l = 0; l < layer1_size; l++) neu1e[l] += grad * syn1_negative[l2 + l];
                                // update syn1_neg 
                                for (l = 0; l < layer1_size; l++) syn1_negative[l2+l] += grad*syn0[l1 + l];
                            }
                        }
                    }
                    // update cur context word which is the input word.
                    if (adagrad) {
                        for (l = 0; l < layer1_size; l++) {
                            float g_t = neu1e[l];
                            syn0_gdsq[l1 + l] += g_t*g_t;
                            syn0[l1 + l] += cur_alpha*g_t*rsqrt(syn0_gdsq[l1+l]);
                        }
                    }else {
                        for (l = 0; l < layer1_size; l++) syn0[l1 + l] += neu1e[l]; 
                    }
            }
        }
        }

    }
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
    vector<long long> words;
    clock_t now; // clock time, all cpu cosume time
    long long word_cnt = 0, pre_word_cnt = 0;
    // read each line 
    int cur_words = 0;
    while (read_line(words, cur_words, fin, fend)) {
        word_cnt += cur_words;
        // shrinkage learning rate every 10k words
        if (word_cnt-pre_word_cnt > 10000) {
            trained_words += word_cnt-pre_word_cnt;
            pre_word_cnt = word_cnt;
            now = clock();
            printf("%cAlpha: %f Progress: %.2f%% Words/thread/sec: %.2fk ", 13, alpha, 
                    static_cast<float>(trained_words)/(iter*total_words+1)*100, 
                    static_cast<float>(trained_words)/(static_cast<float>(now-start+1)/CLOCKS_PER_SEC*1000));
            fflush(stdout);
            if (!adagrad) {
                alpha = start_alpha*(1- static_cast<float>(trained_words)/(iter*total_words+1));
                if (alpha < min_alpha) alpha = min_alpha;
            }
        }
        if (model == "cbow") {
            train_cbow(words, alpha);
        }else if (model == "sk") {
            train_skip_gram(words, alpha);
        } else if (model == "sk-sp") {
            train_skip_gram_with_specializing(words, alpha);
        }
        words.clear();
        cur_words = 0;
    }
    fin.close();
}

void Word2vec::train_model(const string& train_file){
    ifstream fin(train_file, ios::in);
    if (!fin) {
        cerr << "Can't open file " << train_file << endl;
        return;
    }
    fin.close();
    init_network();
    struct timeb train_beg , train_end;
    ftime(&train_beg);
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
    ftime(&train_end);
    int train_cost_sec = train_end.time - train_beg.time;
    cout << "Total training time : " << train_cost_sec/60 << " m "<< train_cost_sec%60 << " s" << endl;
}

void Word2vec::save_vector(const string& output_file) {
    clock_t save_beg, save_end;
    save_beg = clock();
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
    save_end = clock();
    float cost_time = static_cast<float>(save_end-save_beg)/CLOCKS_PER_SEC;
    cout << "Save word vectors into file : " << output_file << "  Cost time : " << cost_time/60 << "m" << endl;
}

