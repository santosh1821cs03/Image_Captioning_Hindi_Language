# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf


class CaptionGenerator(object):
    def __init__(self, word_to_idx, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, n_time_step=16,
                  prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = len(word_to_idx)
        self.L = dim_feature[0] #49
        self.D = dim_feature[1] #2048
        self.M = dim_embed #524
        self.H = dim_hidden #1024
        self.T = n_time_step #19
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)
            
            return h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D]) #(,2048)
            features_proj = tf.matmul(features_flat, w) #(,2048)
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D]) #(,49,2048)
            return features_proj

    # def _attention_layer(self, features, features_proj, h, reuse=False):
    #     with tf.variable_scope('attention_layer', reuse=reuse):
    #         w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
    #         b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
    #         w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

    #         h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
    #         print('h_att',h_att.shape)
    #         out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
    #         print('out_att ',out_att.shape)
    #         alpha = tf.nn.softmax(out_att)
    #         context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
    #         print('context shape',context.shape)
    #         return context, alpha



    # def _attention_layer(self, features, features_proj, h, reuse=False):
    #     with tf.variable_scope('attention_layer', reuse=reuse):
    #         w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
    #         b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
    #         w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

    #         h_att = tf.expand_dims(tf.matmul(h,w), 1)
            
    #         print('h_att',h_att.shape)
    #         out_att = tf.reshape(tf.matmul(h_att,tf.transpose(features_proj,perm=[0,2,1])),[-1,self.L])
    #         print('out_att ',out_att.shape)
    #         # h_att = tf.nn.tanh(tf.expand_dims(tf.matmul(h, w), 1) + b + features_proj) #(N,L,D)
    #         # out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
           	
    #         alpha = tf.nn.softmax(out_att)
    #         context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
    #         print('context shape',context.shape)
    #         return context, alpha

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            # b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            # w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

           
            h_att = tf.matmul(h,w)
            # (M @ v[..., None])[..., 0]
            # print('h_att',h_att.shape)
            
            out_att = tf.matmul(features_proj,h_att[...,None])[...,0]
            # print('out_att ',out_att.shape)
            # h_att = tf.nn.tanh(tf.expand_dims(tf.matmul(h, w), 1) + b + features_proj) #(N,L,D)
            # out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            # print('context shape',context.shape)
            return context, alpha


    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _get_vg(self,features):
        with tf.variable_scope('global'):
                wg = tf.get_variable('wg', [self.D, self.D], initializer=self.weight_initializer)
                bg = tf.get_variable('bg', [self.D], initializer=self.weight_initializer)
                ag = (tf.reduce_sum(features,1))/self.L
                vg = tf.nn.relu(tf.matmul(ag,wg)+bg)
                return vg

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def build_model(self):
        
        features = self.features
        #print(features)
        captions = self.captions
        #print(captions)
        batch_size = tf.shape(features)[0]

        captions_in = captions[:, :self.T]
        #print(captions_in)
        captions_out = captions[:, 1:]
        #print(captions_out)
        mask = tf.to_float(tf.not_equal(captions_out, self._null))
        #print(mask)

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')
        #print(features)
        h = self._get_initial_lstm(features=features)
        #print(c,h)
        x = self._word_embedding(inputs=captions_in)
        #print(x)
        features_proj = self._project_features(features=features)
        vg = self._get_vg(features=features)
        #print(features_proj)
        loss = 0.0
        alpha_list = []
        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=self.H)
        #print(lstm_cell)
         
        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h ,reuse=(t!=0))
            #print(context,alpha)
            alpha_list.append(alpha[:,:])
            
            #print(len(alpha_list))
            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))

            with tf.variable_scope('lstm', reuse=(t!=0)):
                #print(x[:,t,:])
                _, h = gru_cell(inputs=tf.concat( [x[:,t,:], vg],1), state=h)
            
            # context, alpha = self._attention_layer(features, features_proj, h ,reuse=(t!=0))
            # alpha_list.append(alpha[:,:])
            logits = self._decode_lstm(x[:,t,:], h, context, dropout=self.dropout, reuse=(t!=0))

            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions_out[:, t],logits=logits)*mask[:, t] )

        if self.alpha_c > 0:
            print("In Alpha")
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            #print(alphas_all)
            alpha_reg = self.alpha_c * tf.reduce_sum((self.T/self.L - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')

        h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)
        vg = self._get_vg(features=features)
        
        sampled_word_list = []
        alpha_list = []
        beta_list = []
        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            alpha_list.append(alpha[:,:])

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t!=0))
                beta_list.append(beta)
            #print(tf.concat( [x, vg, context],1))
            with tf.variable_scope('lstm', reuse=(t!=0)):
                #print(tf.concat( [x, vg, context],1))
                _, h = gru_cell(inputs=tf.concat( [x, vg],1), state=h)
            
            # context, alpha = self._attention_layer(features, features_proj, h, reuse=(t!=0))
            # alpha_list.append(alpha[:,:])
            logits = self._decode_lstm(x, h, context, reuse=(t!=0))
            #print(logits)
            sampled_word = tf.argmax(logits, 1)
            #print(sampled_word)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2)) # (N, T, L)
        if len(beta_list)==0:
            betas = tf.transpose(tf.squeeze(beta_list))    # (N, T)
        else:
            betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas,sampled_captions
