from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

# WORK_PATH = '/Volumes/Eathoublu/word2vec'
WORK_PATH = '.'

class Word2VecTrainingMaster():

    def __init__(self, new_work=True, corpus_path='', work_path='.', base_name='word2vec.model', num_features=100, min_word_count=1, context=4, auto=True, batch_size=100000, step_save=True):

        self.work_path = work_path
        self.corpus_path = corpus_path
        self.word2vec_model = self.make_word2vec_model(num_features, min_word_count, context)
        self.base_name = base_name
        self.word2vec_model.save(self.work_path + '/' + self.base_name)
        self.step_save = step_save

        if auto:

            with open(self.corpus_path) as f:

                reader = f.readlines()

                f.close()

            length = len(reader)

            model_init = True

            for batch_index in tqdm(range(0, length, batch_size)):

                batch = reader[batch_index: batch_index+batch_size]

                batch = self.batch_compile(batch)

                self.update_model(batch, init=model_init)

                model_init = False

            self.word2vec_model.save(self.work_path + '/' + self.base_name)

    def make_word2vec_model(self, num_features, min_word_count, context):

        return Word2Vec(size=num_features, min_count=min_word_count, window=context)

    def update_model(self, batch, init):

        if self.step_save:

            self.word2vec_model = Word2Vec.load(self.work_path + '/' + self.base_name)

        if init:

            self.word2vec_model.build_vocab(batch)

        else:

            self.word2vec_model.build_vocab(batch, update=True)

        self.word2vec_model.train(batch, total_examples=self.word2vec_model.corpus_count, epochs=self.word2vec_model.iter)

        if self.step_save:

            self.word2vec_model.save(self.work_path + '/' + self.base_name)

    @staticmethod
    def batch_compile(batch):

        sentence_list = []

        for sentence in batch:

            gram_list = sentence.split(' ')

            sentence_list.append(gram_list)

        return sentence_list

if __name__ == '__main__':

    # w2v = Word2VecTrainingMaster(corpus_path='/Volumes/Eathoublu/lang_output/output.txt', num_features=500, work_path=WORK_PATH, batch_size=100, base_name='test.w2v', step_save=False)
    w2v = Word2VecTrainingMaster(corpus_path='mini_20_output.txt', num_features=700,
                                 work_path=WORK_PATH, base_name='2_gram_20_mini_700f_word2vec.model', step_save=False)

# /home/lanyx/anaconda3/bin/python /home/lanyx/20190221/make_word2vec_gen_data.py


