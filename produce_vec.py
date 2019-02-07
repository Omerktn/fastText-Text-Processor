import subprocess
import codecs
from nltk import word_tokenize
import numpy as np
import sys
import h5py

def main():
    reload(sys)
    sys.setdefaultencoding("utf-8")

    fastdir = '/home/user/fastText-0.1.0/'
    cmd = [fastdir + 'fasttext', 'print-sentence-vectors', fastdir + 'vectors.bin']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

    for i in range(len(news_name)):
        dataset = []
        with h5py.File(vecdir + news_name[i] + ".h5", 'w') as hf:
            hf.create_dataset("tw",  data=dataset)                  # create first data file

        for j in range(1,news_count[i]):
            try:
                with codecs.open(newsdir + news_name[i] + '/' + str(j) + '.txt', 'r') as f:
                    text = f.readlines()
                    vecHolder = np.zeros(100)
                    vecCounter = 0

                    for line in text[:-1]:
                        vecCounter = vecCounter + 1
                        p.stdin.write(line)
                        output = p.stdout.readline()
                        output = word_tokenize(output)
                        vecHolder = add(vecHolder,findVec(output))

                    vecHolder = div(vecHolder, vecCounter)
                    dataset.append(vecHolder)
                    f.close
            except:
                pass  # pass if can't open the file

            if (j % 101 == 0):
                sys.stdout.write("\r [" + news_name[i] + "] : " + str(j) + "/" + str(news_count[i]))

        sys.stdout.write("\r [" + news_name[i] + "] : " + str(j) + "/" + str(news_count[i]) + "\n")
        #delete
        with h5py.File(vecdir + news_name[i] + ".h5",  "a") as f:
          del f["tw"]
        #save
        with h5py.File(vecdir + news_name[i] + ".h5", 'w') as hf:
          hf.create_dataset("tw",  data=dataset)


    p.stdin.close()
    p.stdout.close()


def findVec(listText):
    size = len(listText)
    new = np.zeros(100)
    k = 0

    for i in range(size - 100, size):
        new[k] = float(listText[i])
        k = k + 1

    return new

def add(vec1, vec2):
    for i in range(len(vec1)):
        vec1[i] = vec1[i] + vec2[i]
    return vec1

def div(vec, m):
    for i in range(len(vec)):
        vec[i] = vec[i] / m
    return vec


if __name__ == '__main__':
    main()
