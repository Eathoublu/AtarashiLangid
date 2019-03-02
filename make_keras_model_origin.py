from keras.models import *
from keras.layers import *
from gensim.models import Word2Vec
import numpy as np
import re

class VecTrainNnModelMaster(object):

    def __init__(self, w2v_model_path='',
                 language_corpus_path='',
                 language_target_path='',
                 ks_model_name='ks.model',
                 batch_size=32,
                 auto=True,
                 ks_input_dim=300,
                 ks_train_batch=64,
                 ks_train_epoch=30,
                 ks_dense_1=64,
                 ks_dense_2=64,
                 ks_dense_3=8
                 ):

        self.ks_cfg = {'ks_input_dim': ks_input_dim,
                       'ks_train_batch': ks_train_batch,
                       'ks_train_epoch': ks_train_epoch,
                       'ks_dense_1': ks_dense_1,
                       'ks_dense_2': ks_dense_2,
                       'ks_dense_3': ks_dense_3}

        self.ks_model = self.make_ks_model(**self.ks_cfg)

        self.word2vec_model = Word2Vec.load(w2v_model_path)

        self.valid_rule = re.compile('[0-9]+', re.S)

        self.batch_size = batch_size

        self.ks_model_name = ks_model_name

        if auto:

            self.language_src = self.read_file(language_corpus_path)

            self.language_tgt = self.read_file(language_target_path)

            self.data_length = len(self.language_tgt)

            self.ks_train()

    def ks_train(self):

        print(self.data_length, self.batch_size)

        self.ks_model.fit_generator(generator=self.make_batch(batch_size=self.batch_size), epochs=self.ks_cfg['ks_train_epoch'], validation_steps=self.make_batch(batch_size=32, validation=False), steps_per_epoch=self.data_length/self.batch_size)

        self.ks_model.save(self.ks_model_name)

        self.tester()

    def make_batch(self, batch_size, validation=False):

        for times in range(self.ks_cfg['ks_train_epoch']):

            if validation:

                for index in range(0, 1000, batch_size):

                    batch_sentence_list = self.language_src[index: index + batch_size]

                    sentences_vec_list = self.sen2vec(sen_list=batch_sentence_list)

                    target_list = self.language_tgt[index: index + batch_size]

                    target_list = self.tar2one_hot(target_list)

                    # target_list = [[0, 1] for _ in range(len(sentences_vec_list))]

                    yield ({'dense_1_input': np.array(sentences_vec_list)}, {'dense_3': np.array(target_list)})

            else:

                if len(self.language_tgt) != len(self.language_src):

                    raise Exception('LengthError:target length and source length is not same.', str(len(self.language_src))+'!='+str(len(self.language_tgt)))

                for index in range(0, len(self.language_src), batch_size):

                    batch_sentence_list = self.language_src[index: index+batch_size]

                    sentences_vec_list = self.sen2vec(sen_list=batch_sentence_list)

                    target_list = self.language_tgt[index: index+batch_size]

                    target_list = self.tar2one_hot(target_list)

                    # target_list = [[0, 1] for _ in range(len(sentences_vec_list))]

                    yield ({'dense_1_input': np.array(sentences_vec_list)}, {'dense_3': np.array(target_list)})

    def sen2vec(self, sen_list=[]):

        return_list = []

        if not sen_list:

            raise Exception('DataError:No sentence in list.', sen_list)

        for sentence in sen_list:

        #     if self.valid_rule.findall(sentence):
        #
        #         continue

            sentence_vector = np.zeros((self.ks_cfg['ks_input_dim'], ))

            valid_gram = 0

            for gram_index in range(len(sentence)):

                gram = sentence[gram_index: gram_index+2]

                if gram in self.word2vec_model:

                    valid_gram += 1

                    sentence_vector += self.word2vec_model[gram]

            if valid_gram != 0:

                sentence_vector /= valid_gram

            return_list.append(sentence_vector)

        return return_list

    def tar2one_hot(self, target_list):

        # cont = {'es': 0, 'de': 1, 'en': 2, 'vi': 3, 'fr': 4, 'ja': 5, 'zh': 6, 'ko': 7}
        cont = {'de': 0, 'en':1, 'vi':2, 'fr':3, 'es':4, 'ja':5, 'zh':6,
                     'ko':7, 'bs':8, 'ca':9, 'ce':10, 'co':11, 'cs':12, 'cy':13,
                     'da':14, 'el':15, 'eo':16, 'et':17, 'eu':18, 'fa':19}

        one_hot = np.zeros((len(target_list), 20))

        for index in range(len(target_list)):

            one_hot[index, cont[target_list[index][:-1]]] = 1

        return one_hot

    def make_ks_model(self, **kwargs):

        model = Sequential()

        model.add(Dense(kwargs['ks_dense_1'], input_dim=kwargs['ks_input_dim'], activation='relu'))

        model.add(Dropout(0.5))

        model.add(Dense(kwargs['ks_dense_2'], activation='relu'))

        model.add(Dropout(0.5))

        model.add(Dense(kwargs['ks_dense_3'], activation='softmax'))

        model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

        return model

    def tester(self):

        scr_list = self.sen2vec(self.language_src)

        tgt_list = self.tar2one_hot(self.language_tgt)

        _, acc = self.ks_model.evaluate(x=np.array(scr_list), y=np.array(tgt_list), batch_size=32)

        print(acc)

        # print(self.ks_model.predict(np.array(scr_list)))

        # print(self.ks_model.predict(np.array(self.sen2vec(['hello', 'Kiewel moderierte für das ZDF unter anderem den "Fern', "C'est de toute évidence la Société musicale André-Turp qui aura marqué l"]))))

    @staticmethod
    def read_file(file_name, mode='r'):

        return open(file_name, mode=mode).readlines()


if __name__ == '__main__':

    vtnmm = VecTrainNnModelMaster(w2v_model_path='2_gram_20_mini_700f_word2vec.model',
                                  language_corpus_path='200K_rand_lang.txt',
                                  language_target_path='200K_rand_target.txt',
                                  ks_input_dim=700, batch_size=32,
                                  ks_model_name='rand_2_gram_20_700f.ks.model',
                                  ks_train_epoch=10,
                                  ks_dense_3=20
                                  )


# CUDA_VISIBLE_DEVICES=3 nohup /home/lanyx/anaconda3/bin/python  /home/lanyx/20190220/make_keras_model_origin.py
# CUDA_VISIBLE_DEVICES=0 /home/lanyx/anaconda3/bin/python  /home/lanyx/20190221/make_keras_model_origin.py



