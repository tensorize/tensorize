

class Kernel(object):

    def __init__(self, height, width, weights=[]):
        pass

    def initialize(self, f):
        pass

# Stride
class Stride(object):

    def __init__(self, value):
        self.value = value

    def get_value(self, shape):
        return self.value


class FullyConnected(BaseLayer):

    def configure(self, *args, **kwds):
        layer = self.get_previous_layer()
        # get the last dimension
        self.input_tensor = layer.output_tensor
        self.input_dim = self.input_tensor.get_shape().as_list()[-1]

        self.output_dim = kwds['hidden_dim']

        shape = [self.input_dim, self.output_dim]

        # Initialization assumes ReLU activations for now
        # see Delving Deep into Rectifiers: Surpassing Human-LEvel Performance on
        # ImageNet Classification
        # n = float(self.input_dim * self.output_dim)

        # initial = tf.truncated_normal(shape, stddev=tf.sqrt(2.0/n))
        # see Sussillo et al 2014.
        initial = tf.uniform_unit_scaling_initializer(
            factor=1.43, full_shape=shape)

        self.weights = Variable("weights", shape, initial)
        bias_shape = [self.output_dim]
        initial = tf.constant_initializer(value=0.0)

        self.biases = Variable("biases", bias_shape, initial)

        with tf.name_scope('Wx_plus_b'):
            x = self.input_tensor
            W = self.weights.variable
            b = self.biases.variable
            self.preactivate = tf.nn.xw_plus_b(x, W, b)

        self.output_tensor = self.preactivate

    def add_summaries(self):
        self.weights.add_summaries()
        self.biases.add_summaries()
        with tf.name_scope("summaries") as scope:
            tf.histogram_summary(scope + 'pre_activations', self.preactivate)


