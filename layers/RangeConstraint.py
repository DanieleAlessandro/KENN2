import tensorflow as tf


class RangeConstraint(tf.keras.constraints.Constraint):
    """Represent a range constraint for the weight of ClauseEnhancer.

    """

    def __init__(self, min=0, max=500):
        super(RangeConstraint, self).__init__()
        self.min = min
        self.max = max

    def __call__(self, weigth):
        """Clip the weigth value inside the range [min, max]

        :return: the clipped weight
        """
        return tf.clip_by_value(weigth, self.min, self.max)
