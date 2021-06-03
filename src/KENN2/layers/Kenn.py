import tensorflow as tf
from KENN2.layers.residual.KnowledgeEnhancer import KnowledgeEnhancer


class Kenn(tf.keras.layers.Layer):

    def __init__(self, predicates, clauses, activation=lambda x: x, initial_clause_weight=0.5, save_training_data=False, **kwargs):
        """Initialize the knowledge base.

        :param predicates: a list of predicates names
        :param clauses: a list of constraints. Each constraint is a string on the form:
        clause_weight:clause

        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).

        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.

        An example:
           _:nDog,Animal

        """

        super(Kenn, self).__init__(**kwargs)

        self.predicates = predicates
        self.clauses = clauses
        self.activation = activation
        self.initial_clause_weight = initial_clause_weight
        self.knowledge_enhancer = None

    def build(self, input_shape):
        """Build the layer

        :param input_shape: the input shape
        """

        self.knowledge_enhancer = KnowledgeEnhancer(
            self.predicates, self.clauses, self.initial_clause_weight)

        super(Kenn, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Improve the satisfaction level of a set of clauses.

        :param inputs: the tensor containing predicates' pre-activation values for many entities
        :return: final preactivations"""

        deltas, _ = self.knowledge_enhancer(inputs)

        return self.activation(inputs + deltas)

    def get_config(self):
        config = super(Kenn, self).get_config()
        config.update({'predicates': self.predicates})
        config.update({'clauses': self.clauses})
        config.update({'activation': self.activation})
        config.update({'initial_clause_weight': self.initial_clause_weight})
        # config['output_size'] =  # say self. _output_size  if you store the argument in __init__
        return config
