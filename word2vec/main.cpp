/* ############################################################################
* 
* Copyright (c) 2016 ICT MCG Group, Inc. All Rights Reserved
* 
* ###########################################################################
* Brief: 
 
* Authors: zhouxing(@ict.ac.cn)
* Date:    2016/03/26 11:08:28
* File:    main.cpp
*/
#include "word2vec.h"
using namespace std;

// print args info
void info() {
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 10)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-model <int>\n");
    printf("\t\tcbow: Use the continuous bag of words model; sg: for skip-gram model\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");    
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}
int main(int argc, char **argv) {
  if (argc == 1) {
    info();
    return 0;
  }
  int i;
  // set variable
  string train_file = "", output_file = "";
  string save_vocab_file = "", read_vocab_file = "";
  int layer1_size = 100, window = 5;
  string model = "cbow", train_method = "hs";
  float alpha = 0.025, sample = 0.001;
  int num_threads = 10, iter = 1, min_count = 5, negative = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) train_file = string(argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) save_vocab_file = string(argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) read_vocab_file= string(argv[i + 1]);
  if ((i = ArgPos((char *)"-model", argc, argv)) > 0) model = string(argv[i+1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) output_file = string(argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  
  if (model == "cbow") alpha = 0.05;

  if (train_file.empty()) {
  	cerr << "Missing input train file" << endl;
  	return 0;
  }
  if (output_file.empty()) {
  	cerr << "Missing output_file" << endl;
  	return 0;
  }
  Word2vec w2v_model(model, train_method, iter, num_threads, layer1_size, window, negative, min_count, sample, alpha);
  w2v_model.learn_vocab_from_trainfile(train_file);
  w2v_model.train_model(train_file);
  w2v_model.save_vector(output_file);
  return 0;
}
