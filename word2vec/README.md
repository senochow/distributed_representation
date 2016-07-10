# word2vec
Implemention of multi-thread word2vec using c++11.
optimization: Adagrad


Issues:
* forget to reset total words (0), which slow down the learning rate to shrinkage
* trained words caculate error, subsample words is included
* vector index and word index in vocab is not equal
* create huffman tree: create n-1 non-leaf nodes
* know which word to update at each step: update input words vector not itself
* Key points: upadte rule for hs and negtive sample
* adagrad: sum history grads += 

History:
* 0630. fix progress and time bug
    - cause by continue that results word unclear before search for next word.
    - clock is cpu time and unsuitable for global time calculation.
