import glob,os,counter,math,operator, csv
from os.path import relpath

def get_all_files(directory):
    """ files within the given directory.
Param: a string, giving the absolute path to a directory containing data files
Return: a list of relative file paths for all files in the input directory.
    """
    filelist=glob.glob('*.txt')
    relpathlist=[relpath(file) for file in filelist]
    return relpathlist
    
def standardize(rawexcerpt):
    return word_tokenize(rawexcerpt.decode('utf8').lower())

def load_file_excerpts(filepath):
    """that takes an absolute filepath and returns a list of
all excerpts in that file, tokenized and converted to lowercase. Remember that in our data files, each line
consists of one excerpt.
Param: a string, giving the absolute path to a file containing excerpts
Return: a list of lists of strings, where the outer list corresponds to excerpts and the inner lists
correspond to lowercase tokens in that exerpt
    """
    f=open(filepath)
    print 'loading '+filepath
    allexecerpts=[standardize(line)for line in iter(f)]
    print 'number of excerpts in this file is '+str(len(allexecerpts))
    return allexecerpts

def flatten(l):
    return [item for subl in l for item in subl]

def load_directory_excerpts(dirpath):
        """takes an absolute dirpath and returns a list
of excerpts (tokenized and converted to lowercase) concatenated from every file in that directory.
Param: a string, giving the absolute path to a directory containing files of excerpts
Return: a list of lists of strings, where the outer list corresponds to excerpts and the inner lists
correspond to lowercase tokens in that exerpt
        """
        final_list=flatten([load_file_excerpts(files) for files in get_all_files(dirpath)])
        print 'number of excerpts in the entire path is '+str(len(final_list))
        return final_list
