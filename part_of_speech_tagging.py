import sys
import numpy
import tensorflow as tf
import time

USC_EMAIL = 'xinglonb@usc.edu'  # TODO(student): Fill to compete on rankings.
PASSWORD = 'a6bc77e9941dd077'  # TODO(student): You will be given a password via email.
TRAIN_TIME_MINUTES = 11


class DatasetReader(object):

    # TODO(student): You must implement this.
    @staticmethod
    def ReadFile(filename, term_index, tag_index):
        """Reads file into dataset, while populating term_index and tag_index.
     
        Args:
            filename: Path of text file containing sentences and tags. Each line is a
                sentence and each term is followed by "/tag". Note: some terms might
                have a "/" e.g. my/word/tag -- the term is "my/word" and the last "/"
                separates the tag.
            term_index: dictionary to be populated with every unique term (i.e. before
                the last "/") to point to an integer. All integers must be utilized from
                0 to number of unique terms - 1, without any gaps nor repetitions.
            tag_index: same as term_index, but for tags.

        the _index dictionaries are guaranteed to have no gaps when the method is
        called i.e. all integers in [0, len(*_index)-1] will be used as values.
        You must preserve the no-gaps property!

        Return:
            The parsed file as a list of lists: [parsedLine1, parsedLine2, ...]
            each parsedLine is a list: [(termId1, tagId1), (termId2, tagId2), ...] 
        """
        # term0 tag0, term0 tag5
        
        parsed = []
        f = open(filename, 'r')

        if (len(term_index) > 0):
            termid = max(term_index.values())+1
        else:
            termid = 0
        if (len(tag_index) > 0):
            tagid = max(tag_index.values())+1
        else:
            tagid = 0
            
        for line in f:
            parsedLine = []
            for pair in line.strip().split():
                items = pair.rsplit("/", 1)
                term = items[0]
                tag = items[1]
                if term not in term_index:
                    term_index[term] = termid
                    termid += 1
                if tag not in tag_index:
                    tag_index[tag] = tagid
                    tagid += 1
                parsedLine.append((term_index[term], tag_index[tag]))

            parsed.append(parsedLine)
            
        
        return parsed

    # TODO(student): You must implement this.
    @staticmethod
    def BuildMatrices(dataset):
        """Converts dataset [returned by ReadFile] into numpy arrays for tags, terms, and lengths.

        Args:
            dataset: Returned by method ReadFile. It is a list (length N) of lists:
                [sentence1, sentence2, ...], where every sentence is a list:
                [(word1, tag1), (word2, tag2), ...], where every word and tag are integers.

        Returns:
            Tuple of 3 numpy arrays: (terms_matrix, tags_matrix, lengths_arr)
                terms_matrix: shape (N, T) int64 numpy array. Row i contains the word
                    indices in dataset[i].
                tags_matrix: shape (N, T) int64 numpy array. Row i contains the tag
                    indices in dataset[i].
                lengths: shape (N) int64 numpy array. Entry i contains the length of
                    sentence in dataset[i].

            T is the maximum length. For example, calling as:
                BuildMatrices([[(1,2), (4,10)], [(13, 20), (3, 6), (7, 8), (3, 20)]])
            i.e. with two sentences, first with length 2 and second with length 4,
            should return the tuple:
            (
                [[1, 4, 0, 0],    # Note: 0 padding.
                 [13, 3, 7, 3]],

                [[2, 10, 0, 0],   # Note: 0 padding.
                 [20, 6, 8, 20]], 

                [2, 4]
            )
        """
        T = len(max(dataset, key = lambda line: len(line)))
        N = len(dataset)
        terms_matrix = numpy.zeros((N, T), dtype=int)
        for i in range(N):
            numpy.put(terms_matrix[i], list(range(len(dataset[i]))), [pair[0] for pair in dataset[i]])
            
        tags_matrix = numpy.zeros((N, T), dtype=int)
        for i in range(len(dataset)):
            numpy.put(tags_matrix[i], list(range(len(dataset[i]))), [pair[1] for pair in dataset[i]])
            
        lengths = numpy.array([len(line) for line in dataset], dtype=int)
        
        
        return terms_matrix, tags_matrix, lengths

    @staticmethod
    def ReadData(train_filename, test_filename=None):
        """Returns numpy arrays and indices for train (and optionally test) data.

        NOTE: Please do not change this method. The grader will use an identitical
        copy of this method (if you change this, your offline testing will no longer
        match the grader).

        Args:
            train_filename: .txt path containing training data, one line per sentence.
                The data must be tagged (i.e. "word1/tag1 word2/tag2 ...").
            test_filename: Optional .txt path containing test data.

        Returns:
            A tuple of 3-elements or 4-elements, the later iff test_filename is given.
            The first 2 elements are term_index and tag_index, which are dictionaries,
            respectively, from term to integer ID and from tag to integer ID. The int
            IDs are used in the numpy matrices.
            The 3rd element is a tuple itself, consisting of 3 numpy arrsys:
                - train_terms: numpy int matrix.
                - train_tags: numpy int matrix.
                - train_lengths: numpy int vector.
                These 3 are identical to what is returned by BuildMatrices().
            The 4th element is a tuple of 3 elements as above, but the data is
            extracted from test_filename.
        """
        term_index = {'__oov__': 0}  # Out-of-vocab is term 0.
        tag_index = {}
        
        train_data = DatasetReader.ReadFile(train_filename, term_index, tag_index)
        train_terms, train_tags, train_lengths = DatasetReader.BuildMatrices(train_data)
        
        if test_filename:
            test_data = DatasetReader.ReadFile(test_filename, term_index, tag_index)
            test_terms, test_tags, test_lengths = DatasetReader.BuildMatrices(test_data)

            if test_tags.shape[1] < train_tags.shape[1]:
                diff = train_tags.shape[1] - test_tags.shape[1]
                zero_pad = numpy.zeros(shape=(test_tags.shape[0], diff), dtype='int64')
                test_terms = numpy.concatenate([test_terms, zero_pad], axis=1)
                test_tags = numpy.concatenate([test_tags, zero_pad], axis=1)
            elif test_tags.shape[1] > train_tags.shape[1]:
                diff = test_tags.shape[1] - train_tags.shape[1]
                zero_pad = numpy.zeros(shape=(train_tags.shape[0], diff), dtype='int64')
                train_terms = numpy.concatenate([train_terms, zero_pad], axis=1)
                train_tags = numpy.concatenate([train_tags, zero_pad], axis=1)

            return (term_index, tag_index,
                            (train_terms, train_tags, train_lengths),
                            (test_terms, test_tags, test_lengths))
        else:
            return term_index, tag_index, (train_terms, train_tags, train_lengths)


class SequenceModel(object):

    def __init__(self, max_length=310, num_terms=1000, num_tags=40):
        """Constructor. You can add code but do not remove any code.

        The arguments are arbitrary: when you are training on your own, PLEASE set
        them to the correct values (e.g. from main()).

        Args:
            max_lengths: maximum possible sentence length.
            num_terms: the vocabulary size (number of terms).
            num_tags: the size of the output space (number of tags).

        You will be passed these arguments by the grader script.
        """
        self.max_length = max_length
        self.num_terms = num_terms
        self.num_tags = num_tags
        self.x = tf.placeholder(tf.int32, [None, self.max_length], 'x')
        self.lengths = tf.placeholder(tf.int32, [None], 'lengths')

        #variables added by student
        self.tf_session = tf.Session()
        self.learning_rate = tf.placeholder_with_default(numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')
        self.train_op = None
        self.y = tf.placeholder(tf.int32, [None, self.max_length], 'y')
        self.iter = 300
        self.counter = 0
        self.counter2 = 0

    # TODO(student): You must implement this.
    def lengths_vector_to_binary_matrix(self, length_vector):
        """Returns a binary mask (as float32 tensor) from (vector) int64 tensor.
        
        Specifically, the return matrix B will have the following:
            B[i, :lengths[i]] = 1 and B[i, lengths[i]:] = 0 for each i.
        However, since we are using tensorflow rather than numpy in this function,
        you cannot set the range as described.
        """
        #premask = self.tf_session.run(tf.ones([tf.shape(length_vector), self.max_length], dtype=tf.float32))
        lengthsT = tf.expand_dims(length_vector, 1)
        arange = tf.range(0, self.max_length, 1)
        range_row = tf.expand_dims(arange, 0)
        mask = tf.less(range_row, lengthsT)
        
        result = tf.where(mask, tf.ones([tf.shape(length_vector)[0], self.max_length], dtype=tf.float32), tf.zeros([tf.shape(length_vector)[0], self.max_length], dtype=tf.float32))
        return result

    # TODO(student): You must implement this.
    def save_model(self, filename):
        """Saves model to a file."""
        #init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with self.tf_session as sess:
            sess.run(init_op)
            save_path = saver.save(sess, "model.ckpt")

    # TODO(student): You must implement this.
    def load_model(self, filename):
        """Loads model from a file."""
        tf.reset_default_graph()
        saver = tf.train.Saver()
        with self.tf_session as sess:
            saver.restore(sess, "model.ckpt")


    class MinimalRNNCell(tf.keras.layers.Layer):
        def __init__(self, units, **kwargs):
            self.units = units
            self.state_size = units
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='uniform', name='kernel' )
            self.recurrent_kernel = self.add_weight(
                shape = (self.units, self.units),
                initializer = 'uniform',
                name = 'recurrent_kernel')
            self.built = True

        def call(self, inputs, states):
            prev_output = states[0]
            h = tf.keras.backend.dot(inputs, self.kernel)
            output = h + tf.keras.backend.dot(prev_output, self.recurrent_kernel)
            return output, [output]

    # TODO(student): You must implement this.
    def build_inference(self):
        """Build the expression from (self.x, self.lengths) to (self.logits).
        
        Please do not change or override self.x nor self.lengths in this function.

        Hint:
            - Use lengths_vector_to_binary_matrix
            - You might use tf.reshape, tf.cast, and/or tensor broadcasting.
        """
        # TODO(student): make logits an RNN on x.

        #embedding = tf.get_variable('embeddings', [self.num_terms, 30], trainable=True)
        #x_embeddings = tf.nn.embedding_lookup(embedding, self.x)
        
        #keras embedding
        embedding = tf.keras.layers.Embedding(self.num_terms, 40, input_length=self.max_length)
        x_embeddings = embedding(self.x)
        
        cell = tf.keras.layers.SimpleRNNCell(self.num_tags, kernel_regularizer=tf.keras.regularizers.l2(1e-6))#recurrent_dropout = 0.5

        #stacked RNN ((J:0.915, I: 0.917))
        #cl = [tf.keras.layers.SimpleRNNCell(self.num_tags), tf.keras.layers.SimpleRNNCell(self.num_tags), tf.keras.layers.SimpleRNNCell(self.num_tags)]
        #cells = tf.keras.layers.StackedRNNCells([cell, cell])

        #LSTM
        #cell = tf.keras.layers.LSTMCell(self.num_tags)
        #stacked LSTM
        #cells = tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(self.num_tags),
#                                                tf.keras.layers.LSTMCell(self.num_tags),
#                                                tf.keras.layers.LSTMCell(self.num_tags)])
        
        #minimal rnn cell
        #cell = SequenceModel.MinimalRNNCell(self.num_tags)

        #rnn layer
        layer = tf.keras.layers.RNN(cell, return_sequences=True)

        #simple RNN layer
        #layer = tf.keras.layers.SimpleRNN(self.num_tags, return_sequences=True, dropout=0.5)

        # bidirectional
        layer = tf.keras.layers.Bidirectional(layer, merge_mode='concat')

        #LSTM layer # japanese 0.536855
        #layer = tf.keras.layers.LSTM(self.num_tags, return_sequences=True)

        #self.logits = tf.zeros([tf.shape(self.x)[0], self.max_length, self.num_tags])
        self.logits = layer(x_embeddings)

        

        #self.logits = tf.contrib.layers.fully_connected(self.logits, self.num_tags)
        #dense_layer1 = tf.keras.layers.Dense(self.num_tags)
        #self.logits = dense_layer1(self.logits)

        #droppout
        self.logits = tf.keras.layers.Dropout(0.5)(self.logits)
        
        
        #dense
        dense_layer = tf.keras.layers.Dense(self.num_tags, kernel_regularizer=tf.keras.regularizers.l2(1e-6),
                                            )#activity_regularizer=tf.keras.regularizers.l1(0.01)
        self.logits = dense_layer(self.logits)
        self.logits = tf.keras.layers.BatchNormalization(trainable = True)(self.logits)


    # TODO(student): You must implement this.
    def run_inference(self, terms, lengths):
        """Evaluates self.logits given self.x and self.lengths.
        
        Hint: This function is straight forward and you might find this code useful:
        # logits = session.run(self.logits, {self.x: terms, self.lengths: lengths})
        # return numpy.argmax(logits, axis=2)

        Args:
            terms: numpy int matrix, like terms_matrix made by BuildMatrices.
            lengths: numpy int vector, like lengths made by BuildMatrices.

        Returns:
            numpy int matrix of the predicted tags, with shape identical to the int
            matrix tags i.e. each term must have its associated tag. The caller will
            *not* process the output tags beyond the sentence length i.e. you can have
            arbitrary values beyond length.
        """
        #self.tf_session.run(tf.global_variables_initializer())
        logits = self.tf_session.run(self.logits, {self.x:terms, self.lengths:lengths})
        return numpy.argmax(logits, axis=2)

    # TODO(student): You must implement this.
    def build_training(self):
        """Prepares the class for training.
        
        It is up to you how you implement this function, as long as train_on_batch
        works.
        
        Hint:
            - Lookup tf.contrib.seq2seq.sequence_loss 
            - tf.losses.get_total_loss() should return a valid tensor (without raising
                an exception). Equivalently, tf.losses.get_losses() should return a
                non-empty list.
        """

        mask = self.lengths_vector_to_binary_matrix(self.lengths)
        print("mask: ")
        print(mask)
        print("logits: ")
        print(self.logits)
        print("self.y: ")
        print(self.y)
        print("self.num_tags:")
        print(self.num_tags)
        print("self.num_terms:")
        print(self.num_terms)
        print("self.lengths: ")
        print(self.lengths)
        
        #print("loss: ")
        #print(loss)
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.loss = tf.contrib.seq2seq.sequence_loss(logits = self.logits, targets = self.y, weights = mask)
        tf.losses.add_loss(self.loss)
        self.train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)
        
        self.tf_session.run(tf.global_variables_initializer())
        return 0
    #learn_rate=1e-3 batch_size=32
    def train_epoch(self, terms, tags, lengths, batch_size=32, learn_rate=0.000353):
        """Performs updates on the model given training training data.
        
        This will be called with numpy arrays similar to the ones created in 
        Args:
            terms: int64 numpy array of size (# sentences, max sentence length)
            tags: int64 numpy array of size (# sentences, max sentence length)
            lengths:
            batch_size: int indicating batch size. Grader script will not pass this,
                but it is only here so that you can experiment with a "good batch size"
                from your main block.
            learn_rate: float for learning rate. Grader script will not pass this,
                but it is only here so that you can experiment with a "good learn rate"
                from your main block.

        Return:
            boolean. You should return True iff you want the training to continue. If
            you return False (or do not return anyhting) then training will stop after
            the first iteration!
        """
        # <-- Your implementation goes here.
        # Finally, make sure you uncomment the `return True` below.
        # return True
        start_lr = 0.016
        step_lr = 0.004
        if start_lr - (step_lr * self.counter) > learn_rate:
            learn_rate = start_lr - (step_lr * self.counter)
            self.counter += 1
        else:
            learn_rate -= self.counter2*0.000001
            self.counter2 += 1
            
        print(learn_rate)

        #if 26 + (1 * self.counter2) < batch_size:
#            batch_size = 26 + (1 * self.counter2)
#            self.counter2 += 1
            
        self.iter -= 1
        indices = numpy.random.permutation(terms.shape[0])
        loss = 0.
        for si in range(0, terms.shape[0], batch_size):
            se = min(si + batch_size, terms.shape[0])
            slice_x = terms[indices[si:se]] + 0

            #print(self.tf_session.run(tf.contrib.seq2seq.sequence_loss(logits = self.logits, targets = self.y, weights = mask), {self.x: slice_x, self.y: tags[indices[si:se]], self.lengths: lengths[si:se]}))

            
            _, loss = self.tf_session.run([self.train_op, self.loss],
                                {self.x: slice_x,
                                 self.y: tags[indices[si:se]],
                                 self.learning_rate: learn_rate,
                                 self.lengths: lengths[si:se]
                                 })

        #print(loss)

        if self.iter > 0:
            return True
        else:
            return False

    # TODO(student): You can implement this to help you, but we will not call it.
    def evaluate(self, terms, tags, lengths):
        pass



def main_():
    """This will never be called by us, but you are encouraged to implement it for
    local debugging e.g. to get a good model and good hyper-parameters (learning
    rate, batch size, etc)."""
    # Read dataset.
    reader = DatasetReader
    train_filename = sys.argv[1]
    test_filename = train_filename.replace('_train_', '_dev_')
    term_index, tag_index, train_data, test_data = reader.ReadData(train_filename, test_filename)
    (train_terms, train_tags, train_lengths) = train_data
    (test_terms, test_tags, test_lengths) = test_data

    def get_test_accuracy():
        predicted_tags = model.run_inference(test_terms, test_lengths)
        if predicted_tags is None:
            print('Is your run_inference function implented?')
            return 0
        test_accuracy = numpy.sum(
        numpy.cumsum(numpy.equal(test_tags, predicted_tags), axis=1)[numpy.arange(test_lengths.shape[0]),test_lengths-1])/numpy.sum(test_lengths + 0.0)
        return test_accuracy

    model = SequenceModel(train_tags.shape[1], len(term_index), len(tag_index))
    model.build_inference()
    model.build_training()
    #for j in xrange(10):
#        model.train_epoch(train_terms, train_tags, train_lengths)
#        print('Finished epoch %i. Evaluating ...' % (j+1))
#        model.evaluate(test_terms, test_tags, test_lengths)
    MAX_TRAIN_TIME = 3*60 
    start_time_sec = time.clock()
    train_more = True
    num_iters = 0
    while train_more:
        print('  Test accuracy after %i iterations is %f' % (num_iters, get_test_accuracy()))
        train_more = model.train_epoch(train_terms, train_tags, train_lengths)
        train_more = train_more and (time.clock() - start_time_sec) < MAX_TRAIN_TIME
        num_iters += 1

    print('  Final accuracy for is %f' % (get_test_accuracy()))


if __name__ == '__main__':
    main_()

