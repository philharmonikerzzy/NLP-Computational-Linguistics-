def get_tf(sample):
    ''' Param: a list of lowercase strings
Return: a dict with a key for each unique word in the input sample and int values giving the term
frequency for each key
''' 
#     sample=flatten(sample)
    return dict(Counter(sample))

def get_idf(corpus):
    '''Param: a list of lists of lowercase strings
Return: a dict with a key for each unique word in the input corpus and int values giving the IDF
for each key
''' 
    
    unique_words=list(Counter(flatten(corpus)))
    unique_words_ct=[sum(word in sublist for sublist in corpus) for word in unique_words]
    N=len(corpus)
    IDF={word: math.log(float(N)/float(ct)) for word,ct in zip(unique_words, unique_words_ct)}
    print IDF.values()
    with open('tempfiles/IDF_fixed.txt','wb') as handle:
        cPickle.dump(IDF,handle)
    return IDF


def get_tfidf(tf_dict, idf_dict):
    return {word:(tf_dict[word]*idf_dict[word]) for word in tf_dict.keys()}

def get_tfidf_weights_topk(tf_dict,idf_dict,k):
    """that takes TF and IDF dicts as
    output by the functions above, and gives a list of the k words with highest weight and their float
    TF-IDF values in descending order.
    Param: dict, dict, int
    Return: a list of (string, float) tuples of length k"""
    tfidf=get_tfidf(tf_dict,idf_dict)
    """here we are using the interesting object called itemgetter, which lets key command return [1]"""
    return sorted(tfidf.items(),key=operator.itemgetter(1),reverse=True)[:k]  

def get_tfidf_topk(sample,corpus,k):
    """Param: a list of strings; a list of lists of strings; int
Return: a list of (string, float) tuples of length k
    """
    tf_dict=get_tf(sample)
#     idf_dict=get_idf(corpus)
    with open ('tempfiles/IDF_fixed.txt','rb') as handle:
        idf_dict=cPickle.load(handle)
    return get_tfidf_weights_topk(tf_dict,idf_dict,k)


"""From here on are the implementations for problem 2.2 Mutual information"""


def get_word_probs(sample):
    """a list of strings as input, and returns a dict with
each unique word in sample as a key and the estimated probability of that word in sample as the value.
    """
    tf=get_tf(sample)
    return {word:float(count)/float(sum(tf.values())) for word,count in tf.iteritems() }

def get_mi(sample,corpus):
    """generates a dictionary giving the mutual information
between each unique word in sample and the section type represented by sample. Compute p(w) for
each word based on the corpus.
Param: sample is a list of lowercase strings, and corpus is a list of lists of lowercase strings
Return: a dict with string keys and float values
    """
    cond_pw=get_word_probs(sample)
    tf=get_tf(flatten(corpus))
    pw={word:float(count)/float(sum(tf.values())) for word,count in tf.iteritems() }
    return {word:math.log(float(cond_pw[word])/float(pw[word])) for word in cond_pw if tf[word]>4}

def get_mi_topk(sample, corpus,k):
    """Param: list of lowercase strings; list of lists of lowercase tokens; k is an int
Return: a list of (string, float) tuples of length k
    """
    mi_dict=get_mi(sample,corpus)
    print mi_dict
    return sorted(mi_dict.items(),key=operator.itemgetter(1),reverse=True)[:k]

def get_precision(l1,l2):
    print [term in l1 for term in l2]
    return float(sum(term in l1 for term in l2))/float(len(l1))

def get_recall(l1,l2):
    return float(sum(term in l1 for term in l2))/float(len(l2))


def cosine_sim(l1,l2):
    uv_dot=float(sum([el1*el2 for el1,el2 in zip(l1,l2)] ))
    vsize=float(math.sqrt(sum([el1*el1 for el1 in l1])))
    vsize2=float(math.sqrt(sum([el2*el2 for el2 in l2])))
    return uv_dot/(vsize*vsize2)


"""functions needed for part 4"""

def create_feature_space(wordlist):
    '''here the wordlist is the top 1000 words with highest tfidf from the entire corpus'''
    idx=list()
    for i in range(1000):
        idx.append(i)
    feature_dict={word:num for word, num in zip(wordlist,idx)}
    with open('tempfiles/feature_space.txt','wb') as handle:
        cPickle.dump(feature_dict,handle)
    return feature_dict

def vectorize_tfidf(feature_space,idf_dict,sample):
    """
    which takes a feature space (as
output by the previous function), an idf dict (as output by get idf() from Section 2.1) and a sample
as input and returns a vector ~v where each dimension vi of that vector is set to the TF-IDF weight of
the ith word in the feature space for the input sample.
the feature space here is the top 1000 words with tf_idf 
Param: dict; dict; list of strings
Return: list or numpy array
    need to implement a check to prevent new terms from stopping the code
    """
    with open('tempfiles/corpus_tf.txt','rb') as handle:
        corpus_tf=cPickle.load(handle)
    fsp_featured_sample=[word for word in sample if word in feature_space.keys()]
    insample_tf=get_tf(fsp_featured_sample)
    tfidf=get_tfidf(corpus_tf, idf_dict)
    tfidf_insample=get_tfidf(insample_tf,idf_dict)
    vectorized=np.zeros((1,1000))
    for word in fsp_featured_sample:
            vectorized[0,feature_space[word]]=tfidf_insample[word]
    return vectorized
