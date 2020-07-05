from __future__ import division

import tensorflow as tf
from core.modules import *
from tqdm import tqdm


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
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self.num_blocks = 3
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and captions
        self.features = tf.placeholder(tf.float32, [None, self.L, self.D])
        #self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'images')
        self.captions = tf.placeholder(tf.int32, [None, self.T + 1])

    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)
            
            return h

    def _word_embedding(self, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            #x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return w

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('wa', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            #print(features_flat)
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.nn.relu(tf.reshape(features_proj, [-1, self.L, self.D]))
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            wv = tf.get_variable('wv', [self.D, self.D], initializer=self.weight_initializer)
            wg = tf.get_variable('wg', [self.H, self.D], initializer=self.weight_initializer)
            bg = tf.get_variable('bg', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)
            
            h_att = tf.nn.leaky_relu(tf.reshape(tf.matmul(tf.reshape(features_proj, [-1, self.D]),wv), [-1, self.L, self.D]) + tf.expand_dims(tf.matmul(h, wg)+bg, 1))    # (N, L, D)
            
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(features_proj * tf.expand_dims(alpha[:,:], 2), 1, name='context')   #(N, D)
            #context_dash = tf.multiply(tf.expand_dims(alpha[:,-1], 1),st) + tf.multiply((tf.expand_dims((1-alpha[:,-1]),1)),context)
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
            
    def _decode_lstm(self, x, TT, dropout=False, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('logits', reuse=reuse):
            loss = 0
            sampled_word_list =[]
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            h_logits = x
            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            for t in range(TT):
                out_logit = tf.matmul(h_logits[:,t,:], w_out) + b_out
                if t==0 :
                    out_logits = tf.expand_dims(out_logit,1)
                else:
                    out_logits = tf.concat([out_logits,tf.expand_dims(out_logit,1)],1)
                #sampled_word = tf.argmax(out_logits, 1)
                #sampled_word_list.append(sampled_word)
                #x = self._word_embedding(inputs=sampled_word, reuse=True)
                #loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions_out[:,t],logits=out_logits)*mask[:,t] )
            #print("OUT",out_logits)    
            return out_logits
    
    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.99,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))
    
    def decode(self, ys, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs = ys

            # embedding
            #dec = self._word_embedding(inputs=decoder_inputs)  # (N, T2, d_model)
            dec = tf.nn.embedding_lookup(self._word_embedding(), decoder_inputs)  # (N, T2, d_model)
            #dec *= self.hp.d_model ** 0.5  # scale

            dec += positional_encoding(dec, 20)
            dec = tf.layers.dropout(dec, 0.3, training=training)

            # Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=8,
                                              dropout_rate=0.5,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=8,
                                              dropout_rate=0.5,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    dec = ff(dec, num_units=[self.D, self.M])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self._word_embedding()) # (d_model, vocab_size)
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat
    
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
        #dec = self._word_embedding(inputs=captions_in)
        #print(x)
        memory = self._project_features(features=features)
        #vg = self._get_vg(features=features)
        #print(features_proj)
        loss = 0.0
        #print(lstm_cell)
        
        '''for tt in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(tt), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=8,
                                              dropout_rate=0.5,
                                              training=True,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=8,
                                              dropout_rate=0.5,
                                              training=True,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward
                    #print(dec)
                    dec = ff(dec, num_units=[self.H, self.M])
                    #print(dec)
            logits = self._decode_lstm(dec,self.T ,dropout=0.5, reuse=tf.AUTO_REUSE)'''
        logits, preds = self.decode(captions_in, memory)
        print('logits are',logits,captions_out)
        y_ = label_smoothing(tf.one_hot(captions_out, depth=self.V))
        print(y_)
            #loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=captions_out[:,:],logits=logits)*mask[:,:])
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(captions_out, self._null))  # 0: <pad>
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
        return loss

    def build_sampler(self, max_len=20):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')
        memory = self._project_features(features=features)
        #vg = self._get_vg(features=features)
        
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''

        decoder = tf.ones((tf.shape(features)[0], 1), tf.int32) * self._start
        decoder_inputs = decoder
        print("Inference graph is being built. Please be patient.")
        print("decoder",decoder)
        for _ in range(max_len):
            logits, y_hat = self.decode(decoder_inputs, memory, False)
            if tf.reduce_sum(y_hat, 1) == self._null: break
            #print(decoder_inputs,y_hat)
            decoder_inputs = tf.concat((decoder, y_hat), 1)
        print("Yhat",y_hat)
        return y_hat
    
        '''decoder = tf.fill([tf.shape(features)[0],1], self._start)
        for t in range(max_len):
            if t!=0:
                dec = self._word_embedding(inputs=decoder_input)
            else:
                dec = self._word_embedding(inputs=decoder)
            for tt in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(tt), reuse=tf.AUTO_REUSE):
                        # Masked self-attention (Note that causality is True at this time)
                        dec = multihead_attention(queries=dec,
                                                  keys=dec,
                                                  values=dec,
                                                  num_heads=8,
                                                  dropout_rate=0.5,
                                                  training=True,
                                                  causality=True,
                                                  scope="self_attention")

                        # Vanilla attention
                        dec = multihead_attention(queries=dec,
                                                  keys=memory,
                                                  values=memory,
                                                  num_heads=8,
                                                  dropout_rate=0.5,
                                                  training=True,
                                                  causality=False,
                                                  scope="vanilla_attention")
                        ### Feed Forward
                        #print(dec)
                        dec = ff(dec, num_units=[self.H, self.M])
                        print(dec)
            logits = self._decode_lstm(dec,t+1, dropout=0.5, reuse=tf.AUTO_REUSE)
            sampled_word = tf.to_int32(tf.argmax(logits, -1))
            sampled_word_list.append(sampled_word)
            #tf.to_int32(tf.argmax(logits, axis=-1))
            print(sampled_word,decoder,"dec is ",dec)
            decoder_input = tf.concat((decoder, sampled_word), 1)
           
        #sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return sampled_word'''
