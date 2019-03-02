# word2vec+神经网络的语种识别模型 代码说明以及使用方法

1. `make_2_gram.py 用于将训练语料生成n-gram模型，生成后的模型被保存为.pkl文件。`
2. `make_word2vec_gen_data.py 用于训练word2vec模型，使用上一步骤产生的.pkl文件，训练gram的词向量。并将生成的词向量保存为.w2v文件（_gen的意思是分批训练）训练完后，上述产生的gram模型随即被丢弃。`
3. `make_whole_language_target.py 用于生成随机的训练数据（即使几G的数据也没有问题）生成两个.txt文件，一个是数据，一行就是一个句子，另一个是target，一行对应句子的一个标签。`
4. ` make_keras_model_origin.py 构建以及训练模型，送入的数据便是上一步生成的txt，标签示上一步生成的target.txt，训练完成后会生成.ks.model文件`
5. `model_tester.py 测试模型，输入测试语料和target（格式和训练语料相同），训练完成后生成正确率。`
6. 关键的训练代码已经封装成class，只需要简单修改初始化参数即可（例如保存的文件的名字）。