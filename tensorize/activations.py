from __future__ import absolute_import


class RectifiedLinearUnit(BaseLayer):

    def configure(self, alpha=0., max_value=None):
        self.input_tensor = layer.output_tensor
        self.output_tensor = tf.nn.relu(self.input_tensor, name="relu")

    def __apply__(self, tensor):
        self.input_tensor = tensor


ReLU = RectifiedLinearUnit


# Exponential Linear Unit
#
""" Exponential linear unit

# Arguments
    x: Tensor to compute the activation function for.
    alpha: scalar
"""

class Tanh(BaseLayer):

    def configure(self, *args, **kwds):
        layer = self.model.get_previous_layer()
        self.input_tensor = layer.output_tensor
        self.output_tensor = tf.nn.tanh(self.input_tensor, name="tanh")

