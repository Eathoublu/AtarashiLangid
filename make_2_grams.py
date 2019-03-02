import os
from time import time
import re
import pickle as pkl
# from tqdm import tqdm

# DIR = 'source'
# DIR = '../source/'
LIMIT = 200000

DIR_ = '../source/'
# DIR_ = '/Volumes/Eathoublu/pycharm/source/'

if __name__ == '__main__':

    rule = re.compile('[0-9]+|www|http', re.S)
    t1 = time()
   

    sentence_list = []

    file_list = ['de-1M.txt', 'en-1M.txt', 'vi-1M.txt', 'fr-1M.txt', 'es-1M.txt', 'ja-1M.txt', 'zh-2M.txt', 'ko-1M.txt', 'bs-1M.txt', 'ca-1M.txt', 'ceb-1M.txt', 'co-1M.txt', 'cs-1M.txt', 'cy-1M.txt', 'da-1M.txt', 'el-1M.txt', 'eo-1M.txt', 'et-1M.txt', 'eu-1M.txt', 'fa-1M.txt']

    # with open('/Volumes/Eathoublu/lang_output/9_output.txt', 'a') as of:
    with open('mini_20_output.txt', 'a') as of:

        for file in file_list:

            with open(DIR_ + file, 'r') as f:

                reader = f.readlines()

                f.close()

            k = 0

            print('now the', file)

            # for line in tqdm(reader):
            for line in reader:



                if rule.findall(line):

                    continue

                k += 1

                gram_list = []

                for word_index in range(len(line)-2):

                    for gram in line[word_index:word_index+2]:

                        of.write(gram)

                    of.write(' ')

                of.write('\n')

                    # gram_list.append(line[word_index:word_index+3])

                # sentence_list.append(gram_list)

                if k >= LIMIT:

                    break

        print(time() - t1)

         with open('20k_multi_sentence_gram.pkl', 'wb') as of:

             pkl.dump(sentence_list, of)

             of.close()

         print(time() - t1)

         print(sentence_list)

        # 200000 64s

        # 2000000 316s



