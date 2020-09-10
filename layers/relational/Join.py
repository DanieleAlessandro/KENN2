import tensorflow as tf


class Join(tf.keras.layers.Layer):
    """Join layer

    """

    def __init__(self):
        """Initialize the Join layer.

        """
        super(Join, self).__init__()

    def build(self, input_shape):
        """Build the layer

        :param input_shape: the input shape
        """
        super(Join, self).build(input_shape)

    def call(self, unary, binary, index1, index2):
        """Join the unary and binary tensors.

        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param index1: a vector containing the indices of the first object
        of the pair referred by binary tensor
        :param index1: a vector containing the indices of the second object
        of the pair referred by binary tensor
        """

        index1 = tf.squeeze(index1)
        index2 = tf.squeeze(index2)
        return tf.concat([tf.gather(unary, index1, axis=0), tf.gather(unary, index2, axis=0), binary], axis=1)
