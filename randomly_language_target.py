import os
from time import time
import pickle as pkl
# from tqdm import tqdm
from random import choice

# DIR = 'source'
DIR = '../source/'
LIMIT = 100

DIR_ = '../source/'
# DIR_ = '/Volumes/Eathoublu/pycharm/source/'

if __name__ == '__main__':

    t1 = time()

    walker = os.walk(DIR)

    # of = open('/Volumes/Eathoublu/lang_output/output.txt', 'wb')

    file_list = []

    for i in walker:

        file_list = i[2]

        break

    sentence_list = []

    tf = open('10_rand_target.txt', 'a')

    # file_list = ['de-1M.txt', 'en-1M.txt', 'vi-1M.txt', 'fr-1M.txt', 'es-1M.txt', 'ja-1M.txt', 'zh-2M.txt', 'ko-1M.txt']
    file_list = ['de-1M.txt', 'en-1M.txt', 'vi-1M.txt', 'fr-1M.txt', 'es-1M.txt', 'ja-1M.txt', 'zh-2M.txt', 'ko-1M.txt', 'bs-1M.txt', 'ca-1M.txt', 'ceb-1M.txt', 'co-1M.txt', 'cs-1M.txt', 'cy-1M.txt', 'da-1M.txt', 'el-1M.txt', 'eo-1M.txt', 'et-1M.txt', 'eu-1M.txt', 'fa-1M.txt']

    f_reader = []

    for file in file_list:

        f_reader.append([file[:2], open(DIR_ + file).readlines(), 0])

    with open('10_rand_lang.txt', 'a') as of:

        while True:

            next_sentence = choice(f_reader)

            if next_sentence[2] >= LIMIT or next_sentence[2] >= len(next_sentence[1]):

                f_reader.pop(f_reader.index(next_sentence))

                if not f_reader:

                    break

                continue


            of.write(next_sentence[1][next_sentence[2]])

            next_sentence[2] += 1

            tf.write(next_sentence[0] + '\n')

            flag = False

            for i in f_reader:

                if i[2] <= LIMIT:

                    flag = True

            if not flag:

                break

        tf.close()

        of.close()

        print(time() - t1)



