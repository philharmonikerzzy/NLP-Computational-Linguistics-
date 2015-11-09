from nltk import word_tokenize
import glob,os,counter,math,operator, csv
from os.path import relpath
from collections import Counter
import cPickle
import numpy as np

def get_section_representations(dirname,idf_dict,feature_space):
    """computes
a representation for each section.txt file in dirname based on TF-IDF weight.
Param: string; dict; dict
Return: a dict of section representations, where each key corresponds to the name of a file in dirname
and the value is a list or numpy array
    """
    allfiles_list=get_all_files(dirname)
    rep_dict= {filename:vectorize_tfidf(feature_space,idf_dict,flatten(load_file_excerpts(filename))) for filename in allfiles_list}
    with open('tempfiles/representation_dict.txt','wb') as handle:
        cPickle.dump(rep_dict,handle)
    return rep_dict

    
def predict_class(excerpt,representation_dict,feature_space,idf_dict):
    excerpt_vec=vectorize_tfidf(feature_space,idf_dict,excerpt)
    cos_sims={cat:cosine_sim(excerpt_vec.tolist()[0],representation_dict[cat].tolist()[0]) for cat in representation_dict}
    return sorted(cos_sims.items(),key=operator.itemgetter(1),reverse=True)[0]

def label_sents(excerptfile,outputfile):
    """takes as input a excerptfile containing
new sentences to be labeled (one per line), and outputs predicted section class labels to the outputfile,
one per line, as below
    """    
    input_raw=load_file_excerpts(excerptfile)
    print input_raw
    with open ('tempfiles/feature_space.txt','rb') as handle:
        feature_space=cPickle.load(handle)
    with open ('tempfiles/IDF_fixed.txt','rb') as handle:
        idf_dict=cPickle.load(handle)
    with open ('tempfiles/representation_dict.txt','rb') as handle:
        rep_dict=cPickle.load(handle)
    outlist=[predict_class(excerpt, rep_dict,feature_space,idf_dict)[0] for excerpt in input_raw]
    with open('tempfiles/'+outputfile,'w') as f:
         writer=csv.writer(f)
         for label in outlist:
             writer.writerow([label[:-4]])
             
             

def prepare_cluto_tfidf(samplefile, labelfile, matfile,corpus):
    with open ('tempfiles/IDF_fixed.txt','rb') as handle:
        idf_dict=cPickle.load(handle)
    input=load_file_excerpts(samplefile)
    print len(input)
    corpus_tf=get_tf(flatten(corpus))
    total_topk_tfidf_wordlist=[tuple[0] for tuple in get_tfidf_weights_topk(corpus_tf, idf_dict, 1000)]
    feature_sp=create_feature_space(total_topk_tfidf_wordlist)
    vectorized_list=[vectorize_tfidf(feature_sp,idf_dict,sample) for sample in input]
    with open('vcluster/'+labelfile,'wb') as handle:
        writer=csv.writer(handle)
        for words in feature_sp.keys():
            writer.writerow([words])
     
    with open('vcluster/'+matfile,'wb') as f:
        writer=csv.writer(f, delimiter=" ")
        f.write(str(len(samplefile))+' '+str(1000)+'\r\n')
#         writer.writerow([len(samplefile)]+[1000])
        for vector in vectorized_list:
            f.write(' '.join(str(num) for num in vector[0].tolist()))
            f.write('\r\n')
#             print vector[0].tolist() 
#             writer.writerow([(vector[0]).tolist()])
                
    return


def prepare_cluto_mi(samplefile, labelfile, matfile,corpus):
    with open ('tempfiles/IDF_fixed.txt','rb') as handle:
        idf_dict=cPickle.load(handle)
    input=load_file_excerpts(samplefile)
    corpus_tf=get_tf(flatten(corpus))
    total_topk_tfidf_wordlist=[tuple[0] for tuple in get_tfidf_weights_topk(corpus_tf, idf_dict, 1000)]
    feature_sp=create_feature_space(total_topk_tfidf_wordlist)
    print "start getting corpus word probilities"
    word_probs=get_word_probs(flatten(corpus))
    vectorized_list=[vectorize_mi(feature_sp,word_probs,sample) for sample in input]
    with open('vcluster/'+labelfile,'wb') as handle:
        writer=csv.writer(handle)
        for words in feature_sp.keys():
            writer.writerow([words])
     
    with open('vcluster/'+matfile,'wb') as f:
        writer=csv.writer(f, delimiter=" ")
        f.write(str(len(samplefile))+' '+str(1000)+'\r\n')
#         writer.writerow([len(samplefile)]+[1000])
        for vector in vectorized_list:
            f.write(' '.join(str(num) for num in vector[0].tolist()))
            f.write('\r\n')
#             print vector[0].tolist() 
#             writer.writerow([(vector[0]).tolist()])
                
    return




def generate_mi_feature_labels(dirname,k,corpus):
    """
    takes a directory,
an integer, and a list of excerpts, and returns a list containing the union of the sets of words returned
by get mi topk(file, corpus, k) for each file in dirname.
Param: string; int; list of lists
Return: list
    """
    tuples_from_files=flatten([get_mi_topk(flatten(load_file_excerpts(filename)),corpus,k) for filename in get_all_files(dirname)])
    topk_set=set([tuple1[0] for tuple1 in tuples_from_files])
    return topk_set

def vectorize_mi(feature_space,word_probs,sample):
     fsp_featured_sample=[word for word in sample if word in feature_space.keys()]
     insample_cond_pw=get_word_probs(fsp_featured_sample)
     insample_mi_dict={word:math.log(insample_cond_pw[word]/word_probs[word]) for word in insample_cond_pw}
     vectorized=np.zeros((1,1000))
     for word in fsp_featured_sample:
            vectorized[0,feature_space[word]]=insample_mi_dict[word]
     return vectorized
