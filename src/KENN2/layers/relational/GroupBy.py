import tensorflow as tf


class GroupBy(tf.keras.layers.Layer):
    """GroupBy layer

    """

    def __init__(self, number_of_unary_predicates):
        """Initialize the GroupBy layer.

        """
        super(GroupBy, self).__init__()
        self.n_unary = number_of_unary_predicates

    def build(self, input_shape):
        """Build the layer

        :param input_shape: the input shape
        """
        super(GroupBy, self).build(input_shape)

    def call(self, unary, binary, deltas, index1, index2):
        """Split the deltas matrix in unary and binary deltas.

        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param deltas: the tensor containing the delta values
        :param index1: a vector containing the indices of the first object
        of the pair referred by binary and deltas tensors
        :param index2: a vector containing the indices of the second object
        of the pair referred by binary and deltas tensors
        """
        ux = deltas[:, :self.n_unary]
        uy = deltas[:, self.n_unary:2 * self.n_unary]
        b = deltas[:, 2 * self.n_unary:]
        shape = tf.shape(unary)

        return tf.scatter_nd(index1, ux, shape) + tf.scatter_nd(index2, uy, shape), b
