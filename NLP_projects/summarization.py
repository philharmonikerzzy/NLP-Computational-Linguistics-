import glob
import os,operator,copy,math
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import wordnet as wn
import subprocess
import re
from collections import Counter

def standardize(rawexcerpt):
    return word_tokenize(rawexcerpt.decode('utf8'))

def get_tf(sample):
    return dict(Counter(sample))

def load_file_excerpts(filepath):
    """that takes an absolute filepath and returns a list of
all excerpts in that file, tokenized and converted to lowercase. Remember that in our data files, each line
consists of one excerpt.
Param: a string, giving the absolute path to a file containing excerpts
Return: a list of lists of strings, where the outer list corresponds to excerpts and the inner lists
correspond to lowercase tokens in that exerpt
    """
    print 'loading '+filepath
    f=open(filepath)

    allexecerpts=[standardize(line)for line in iter(f)]
    print 'number of excerpts in this file is '+str(len(allexecerpts))
    return allexecerpts


def get_all_files(directory):
    """ files within the given directory.
Param: a string, giving the absolute path to a directory containing data files
Return: a list of relative file paths for all files in the input directory.
    """
    filelist=glob.glob(directory+'/*')
    return filelist

def get_subdir(dir):
    return [os.path.join(dir,subdir) for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir,subdir))]

def flatten(l):
    return [item for subl in l for item in subl]


def generate_config_files(template_file,outdir,indir):
    dir_list=get_subdir(indir)
    with open(template_file,'r') as file:
        template=file.readlines()
        for i in range(len(dir_list)):
            g=open(outdir+'/config_'+str(i),'wb')
            template[-5]='inputDir = '+str(dir_list[i])
            template[-1]='outputFile = '+outdir+'/dev_out/dev_'+str(i)+'.ts'
            g.writelines(template)
        
    return

def load_topic_words(topic_file,n):
    """
    given a .ts file and integer n, returns a list
    of the top-n topic words from that file in order of descending 
    2
    statistic.
    Param: topic file is a str that gives the path to a .ts file as output by the TopicWords tool, and
    n is an int.
    Return: A list of strings length n.
    """
    with open(topic_file,'r') as file:
        topic_list=file.readlines()
        word_prob={word.split()[0]: float(word.split()[1]) for word in topic_list}
        sorted_topic=sorted(word_prob.items(),key=operator.itemgetter(1),reverse=False)[:n]
    return [sorted_tuple[0] for sorted_tuple in sorted_topic]

def load_all_topic_words(topic_file):
    with open(topic_file,'r') as file:
        topic_list=file.readlines()
        word_prob={word.split()[0]: float(word.split()[1]) for word in topic_list}
        sorted_topic=sorted(word_prob.items(),key=operator.itemgetter(1),reverse=False)
    return [sorted_tuple[0] for sorted_tuple in sorted_topic if sorted_tuple[1]>10]


def recursive_cluster(keylist,cluster_main):
    """
    recursively checking if each element of the keylist has any overlapping lemmas or lemmas of hypernyms and hyponyms with the rest of the words in the list
    the base case is when keylist is empty
    
    """
    if not keylist:
        return cluster_main
    else:
        for word in keylist:
          if word not in cluster_main:  #checking to see if word is already in the cluster
            cluster=[word]  # if not, form temporary cluster with the word itself
            word_syn=wn.synsets(word)
            word_hypos=flatten([ syn.hyponyms() for syn in word_syn])
            word_hyper=flatten([syn.hypernyms() for syn in word_syn])
            word_lemmas=set(flatten([syn.lemmas() for syn in word_syn]))
            word_hypo_lemmas=set(flatten([hypo.lemmas() for hypo in word_hypos]))
            word_hyper=set(flatten([hyper.lemmas() for hyper in word_hyper]))
            word_lemmas_union=set.union(word_lemmas,word_hypo_lemmas,word_hyper)
            keylist_copy=copy.deepcopy(keylist) #make a temp keylist without the first element
            keylist_copy.remove(keylist_copy[0])
            for i in range(1,len(keylist)):
                target_lemmas=set(flatten([syn.lemmas() for syn in wn.synsets(keylist[i])]))
                target_hypos=flatten([syn.hyponyms() for syn in wn.synsets(keylist[i])])
                target_hyper=flatten([syn.hypernyms() for syn in wn.synsets(keylist[i])])
                target_hypo_lemmas=set(flatten([hypo.lemmas() for hypo in target_hypos]))
                target_hyper_lemmas=set(flatten([hyper.lemmas() for hyper in target_hyper]))
                target_lemmas_union=set.union(target_lemmas,target_hyper_lemmas,target_hypo_lemmas)
                if set.intersection(word_lemmas_union,target_lemmas_union):
                    cluster.append(keylist[i])
                    keylist_copy.remove(keylist[i])
            print cluster
            print keylist
            print keylist_copy        
            cluster_main.append(cluster) #add the temporary cluster to the main cluster
            return recursive_cluster(keylist_copy,cluster_main) # run the same method with a smaller keylist and a larger main cluster 

def cluster_keywords_wn(keylist, outputfile):
    """
    Param: list of str, str
Return: none
Grading:You should turn in the outputile generated by running cluster keywords wn on the top-20 topic
words from dev 10 as hw4 2 1.txt. We will grade the accuracy of your submission.

    """
    f=open(outputfile,'w')
    cluster_list=recursive_cluster(keylist,[])
    line_list=[",".join(subl) for subl in cluster_list ]
    for line in line_list:
        print line
        f.write(str(line)+'\n')
    f.close()
    return   

def expand_keywords_dp(keylist,input_dir,outputfile):
    raw_parses=subprocess.check_output(" /home1/c/cis530/hw4/lexparser.sh "+input_dir,shell=True)
#     raw_parses=subprocess.check_output(" /home1/c/cis530/hw4/lexparser.sh '/home1/c/cis530/hw4/dev_input/dev_10/*'",shell=True)
    raw_parses=raw_parses.split()
    f=open(outputfile,'wb')
    for word in keylist:
        f.write(word+'\t')
        temp_list=[]
        for idx,parse in enumerate(raw_parses):
            if word in parse:
                if parse[-1]==')':
                    temp_list.append(raw_parses[idx-1][re.search('\(',raw_parses[idx-1]).end():re.search('-[0-9]+',raw_parses[idx-1]).start()])
                else:
                    temp_list.append(raw_parses[idx][re.search('\(',raw_parses[idx]).end():re.search('-[0-9]+',raw_parses[idx]).start()])        
        temp_list=list(set(temp_list))
        if word in temp_list:
            
            temp_list.remove(word)         
        f.write(','.join(map(str,temp_list)))
        f.write('\n')
    return

def summarize_baseline(directory,outputfile):
    filelist=glob.glob(directory+'/*')
    word_count=0
    file_idx=0;
    g=open(outputfile,'wb')
    while word_count<100:
        f=open(filelist[file_idx])
        first_sent=f.readline()
        g.write(first_sent)
        word_count=word_count+len(first_sent.strip().split())
        file_idx=file_idx+1
    return

def build_nigram_model(filelist):
    input_corpus=flatten([load_file_excerpts(file) for file in filelist])
    with open('/home1/c/cis530/hw4/stopwords.txt') as f:
        stop_words=f.readlines()
    stop_words=[word.strip() for word in stop_words]
    word_count=dict(Counter(flatten(input_corpus)))
    filtered_counts={word.encode('utf8'):float(word_count[word]) for word in word_count.keys() if word.encode('utf8') in stop_words}
    unigram_prob={word:filtered_counts[word]/float(len(filtered_counts))for word in filtered_counts.keys()}
    return unigram_prob 

def summarize_kl(inputdir,outputfile):
    filelist=get_all_files(inputdir)
    unigram_prob=build_nigram_model(filelist)
    sent_list=[]
    with open('/home1/c/cis530/hw4/stopwords.txt') as f:
        stop_words=f.readlines()
    stop_words=[word.strip() for word in stop_words]
    for file in filelist:
        with open(file) as f:
            sent_list.append([sent.strip() for sent in f.readlines()])
            
    sent_list=flatten(sent_list) 
    summary=[]
    word_count=0
    while word_count<=100:
        """
        next step is to get a list of all sentences from all the excerpts

        """
        KL_list=[]
        for idx, sent in enumerate(sent_list):

            summary_temp=copy.deepcopy(summary)
            summary_temp.append(sent)
            tokenized_summary=flatten([word_tokenize(sent) for sent in summary_temp])
            summary_count=dict(Counter(tokenized_summary))
            filtered_s_count={word:float(summary_count[word]) for word in summary_count.keys() if word in stop_words}
            summary_prob={word:float(filtered_s_count[word])/float(len(filtered_s_count)) for word in filtered_s_count.keys()}
            KL_temp=[float(summary_prob[word]*math.log(float(summary_prob[word])/float(unigram_prob[word]))) for word in filtered_s_count.keys() if word in stop_words]
            KL_new=sum(KL_temp)
            KL_list.append(((idx,KL_new)))
        print KL_list    
        min_idx=sorted(KL_list,key=operator.itemgetter(1))[0][0]
        print 'the new sentence incorporated is '+str((min_idx,KL_list[min_idx]))
        summary.append(sent_list[min_idx])
        print len(sent_list[min_idx].strip().split())
        word_count=word_count+len(sent_list[min_idx].strip().split())
        del sent_list[min_idx]

    g=open(outputfile,'w')
    for sent in summary:
        g.write(sent)
        g.write('\n')
    g.close()
    return
# 
#keylist=load_topic_words('hw4/dev_10.ts', 20)
#cluster=recursive_cluster(keylist, [])
#cluster_keywords_wn(keylist, 'hw4/hw4_2_1.txt')
