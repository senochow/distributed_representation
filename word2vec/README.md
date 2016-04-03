# word2vec
Implemention of multi-thread word2vec using c++11.
optimization: Adagrad


List:
* forget to reset total words (0), which slow down the learning rate to shrinkage
* trained words caculate error, subsample words is included
* vector index and word index in vocab is not equal
* create huffman tree: create n-1 non-leaf nodes
* know which word to update at each step: update input words vector not iteself
* Key points: upadte rule for hs and negtive sample

