import tensorflow as tf
import numpy as np

flags = tf.flags
flags.DEFINE_string('test','111','oifpewijfijwaf')
flags.DEFINE_integer('number', 10, 'iaowfpjiajfe')
FLAGS = flags.FLAGS

class word_embedding():
    def __init__(self, len, batch_size):
        self.embedding_size = 200
        self.len = len
        self.batch_size = batch_size
        self.cell = 512
        self.output_size = 1
        self.input = tf.placeholder(shape=[self.batch_size, self.len], dtype=tf.int32, name='input')
        self.pre_label = self.placeholder([self.batch_size, self.len, 1], 'pre_label')
        self.label = self.placeholder([self.batch_size, self.output_size], 'label')
        self.time_step = self.placeholder([self.batch_size], 'time_step')
        self.create_embedding()
        self.pre_train()
        self.lstm_model()

    def placeholder(self, shape, name):
        return tf.placeholder(shape=shape, dtype='float32', name=name)

    def create_embedding(self):
        self.word2embedding = self.init_variable([3000, self.embedding_size])
        self.emb_data = tf.nn.embedding_lookup(self.word2embedding, self.input)
        #self.data2 = tf.nn.embedding_lookup(self.word2embedding, self.input2)

    def init_variable(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype='float32')

    def pre_train(self):
        w = self.init_variable([3000, self.embedding_size])
        b = self.init_variable([3000])
        self.words = tf.reshape(self.emb_data, shape=[-1, self.embedding_size])
        self.pre_label_1 = tf.reshape(self.pre_label, shape=[-1, 1])
        self.pre_loss = tf.reduce_mean(tf.nn.nce_loss(w, b, self.pre_label_1, self.words, num_sampled=100, num_classes=3000))
        opt = tf.train.GradientDescentOptimizer(0.01)
        pre_var_list = [w, b, self.word2embedding]
        grad_list = tf.gradients(self.pre_loss, pre_var_list)
        grad_list, _ = tf.clip_by_global_norm(grad_list, clip_norm=5)
        self.pre_train_op = opt.apply_gradients(zip(grad_list, pre_var_list))

    def lstm_model(self):
        weight = self.init_variable([self.cell, self.output_size])
        bias = self.init_variable([self.output_size])
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell, forget_bias=0, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, 0.8)
        cells = tf.nn.rnn_cell.MultiRNNCell([cell]*1, state_is_tuple=True)
        self.init_state = cells.zero_state(self.batch_size, tf.float32)
        output, hidden = tf.nn.dynamic_rnn(cells, self.emb_data, sequence_length=self.time_step, initial_state=self.init_state)
        result = tf.nn.sigmoid(tf.add(tf.matmul(hidden[0][1], weight), bias))
        self.loss = tf.reduce_mean((result - self.label)**2)
        self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

def prepare_data(batch_size):
    import nltk
    import re
    english = 'train.csv'
    import pandas as pd
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    wordnet_lemmatizer = WordNetLemmatizer()
    import collections

    df = pd.read_csv(english, encoding='utf-8')
    txt = df.txt.values
    emotion = df.Label.values
    #
    txt = txt[:200]
    emotion = emotion[:200]
    #
    emotion = np.reshape(emotion, newshape=[-1, 1])
    words = []
    sentenses = []
    for line in txt:
        content = nltk.word_tokenize(line)
        content = [wordnet_lemmatizer.lemmatize(cont) for cont in content]
        content = [word for word in content if word not in stopwords.words('english')]
        sentenses.append(content)
        for word in content:
            words.append(word)
    word_dict = collections.Counter(words)
    word_dict = word_dict.most_common(3000)
    dict = {}
    i = 0
    for key, count in word_dict:
        dict[key] = i
        i = i + 1

    max_len = 0
    data_len = []
    all_data = []
    all_label = []
    train_words = []
    label_words = []
    for sentense in sentenses:
        data = []
        label = []
        for i in range(len(sentense) - 1):
            data_tmp = dict.get(sentense[i])
            label_tmp = dict.get(sentense[i + 1])
            if data_tmp and label_tmp:
                data.append(data_tmp)
                label.append(label_tmp)

                train_words.append(data_tmp)
                label_words.append(label_tmp)
        data_len.append(len(data))
        if max_len < len(data):
            max_len = len(data)
        all_data.append(data)
        label = np.reshape(label, newshape=[len(label), 1])
        all_label.append(label)

    pad_data = np.zeros([len(all_data), max_len])
    for i, tmp in enumerate(all_data):
        pad_data[i][:data_len[i]] = tmp
    length = len(all_label)

    pre_train = []
    pre_label = []
    for i in range(len(train_words)//(max_len*batch_size)):
        pre_train.append(np.reshape(train_words[i*max_len*batch_size:(i+1)*max_len*batch_size], newshape=[batch_size, max_len]))
        pre_label.append(np.reshape(label_words[i * max_len * batch_size:(i + 1) * max_len * batch_size], newshape=[batch_size, max_len, 1]))

    train_data = []
    label = []
    time_step = []
    num = length // batch_size
    for i in range(num):
        train_data.append(pad_data[i*batch_size:(i+1)*batch_size])
        label.append(emotion[i*batch_size:(i+1)*batch_size])
        time_step.append(data_len[i*batch_size:(i+1)*batch_size])
    return train_data, pre_train, pre_label, label, time_step, max_len

def main(argv=None):
    FLAGS.test = 'test'
    FLAGS.num = 10
    batch_size =10
    train_data, pre_train, pre_label, label, time_step, max_len = prepare_data(batch_size)
    print('data')
    print(np.shape(train_data))
    print(np.shape(label))
    print(np.shape(time_step))

    print('pre')
    print(np.shape(pre_train))
    print(np.shape(pre_label))

    with tf.Session() as sess:
        model = word_embedding(max_len, batch_size)
        init = tf.global_variables_initializer()
        sess.run(init)
        #pre train
        for i in range(100):
            for x, y in zip(pre_train, pre_label):
                pre_dict = {model.input:x, model.pre_label:y}
                pre_loss, _ = sess.run([model.pre_loss, model.pre_train_op], feed_dict=pre_dict)
                print('pre train loss:', pre_loss)
        #train
        for i in range(1000):
            for x, y, z in zip(train_data, label, time_step):
                dict = {model.input:x, model.label:y, model.time_step:z}
                loss, _ = sess.run([model.loss, model.train_op], feed_dict=dict)
                print('train loss:', loss)

if __name__ == '__main__':
    tf.app.run()