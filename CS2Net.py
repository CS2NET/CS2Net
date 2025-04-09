import tensorflow as tf
from tensorflow.contrib import layers


class SelfAttention:
    """SelfAttention class"""

    def __init__(self,
                 dim,
                 key_mask,
                 query_mask,
                 length,
                 M=None):

        self.key_mask = key_mask
        self.query_mask = query_mask
        self.length = length
        self.dim = dim
        self.M = M

    def build(self, inputs, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            output = self.scaled_dot_product(inputs, inputs, inputs)
            return output

    def scaled_dot_product(self, qs, ks, vs):

        o1 = tf.matmul(qs, ks, transpose_b=True)

        if self.M is not None:
            M = tf.expand_dims(self.M, axis=1)
            o1 = o1 + M
        o2 = o1 / (self.dim ** 0.5)

        if self.key_mask is not None:
            padding_num = -2 ** 32 + 1
            mask = tf.expand_dims(self.key_mask, 1)
            mask = tf.tile(mask, [1, self.length, 1])
            paddings = tf.ones_like(o2) * padding_num
            o2 = tf.where(tf.equal(mask, 0), paddings, o2)

        o3 = tf.nn.softmax(o2)

        if self.query_mask is not None:
            mask = tf.expand_dims(self.query_mask, 2)
            o3 = o3 * tf.cast(mask, tf.float32)

        return tf.matmul(o3, vs)



class FlipGradientBuilder(object):
    '''Gradient Reversal Layer '''

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


class CS2NET:
    def __init__(self):
        flip_gradient = FlipGradientBuilder(）
            

    def build(self, scope):

        def diff_loss(feat1, feat2, names): 

            correlation_matrix = tf.matmul(feat1, feat2, transpose_b=True)
    
            correlation_matrix = tf.reshape(correlation_matrix,[-1,1])
    
            cost = tf.reduce_sum(correlation_matrix)
            cost = tf.where(cost > 0, cost, 0, name='value')
    
            assert_op = tf.Assert(tf.is_finite(cost), [cost])
            with tf.control_dependencies([assert_op]):
                loss_diff = tf.identity(cost,name=names)
    
            return loss_diff
            

        with tf.variable_scope('logit'):
        
            input = tf.concat([no_diff_feats_input, id_fea], axis=1)
    
            hidden1 = layers.fully_connected(input, 256, activation_fn=activation_fn,
                                                   scope='hidden1',
                                                   variables_collections=[dnn_parent_scope])
            hidden2 = layers.fully_connected(hidden1, 128, activation_fn=activation_fn,
                                                   scope='hidden2',
                                                   variables_collections=[dnn_parent_scope])
            hidden3 = layers.fully_connected(hidden2, 64, activation_fn=activation_fn,
                                                   scope='hidden3',
                                                   variables_collections=[dnn_parent_scope])


            treatment_hdiden1 = layers.fully_connected(input, 256, activation_fn=activation_fn,
                                               scope='treatmenthdiden1',
                                               variables_collections=[dnn_parent_scope])
            treatment_hdiden2 = layers.fully_connected(treatment_hdiden1, 128, activation_fn=activation_fn,
                                                   scope='treatmenthdiden2',
                                                   variables_collections=[dnn_parent_scope])
            treatment_hdiden3 = layers.fully_connected(treatment_hdiden2, 64, activation_fn=activation_fn,
                                                   scope='treatmenthdiden3',
                                                   variables_collections=[dnn_parent_scope])
            
            factual_hidden1 = layers.fully_connected(input, 256, activation_fn=activation_fn,
                                                   scope='factualhidden1',
                                                   variables_collections=[dnn_parent_scope])
            factual_hidden2 = layers.fully_connected(factual_hidden1, 128, activation_fn=activation_fn,
                                                   scope='factualhidden2',
                                                   variables_collections=[dnn_parent_scope])
            factual_hidden3 = layers.fully_connected(factual_hidden2, 64, activation_fn=activation_fn,
                                                   scope='factualhidden3',
                                                   variables_collections=[dnn_parent_scope])


            diff_loss1 = diff_loss(dragonhidden3,treatment_hdiden3, 'dt')
            diff_loss2 = diff_loss(dragonhidden3,factual_hidden3,'df')
            diff_loss3 = diff_loss(treatment_hdiden3,factual_hidden3, 'tf')
            
        
