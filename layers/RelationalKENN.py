import tensorflow as tf
from layers.residual.KnowledgeEnhancer import KnowledgeEnhancer
from layers.relational.Join import Join
from layers.relational.GroupBy import GroupBy


class RelationalKENN(tf.keras.layers.Layer):

    def __init__(self, unary_predicates, binary_predicates, unary_clauses, binary_clauses, activation=lambda x: x, initial_clause_weight=0.5, **kwargs):
        """Initialize the knowledge base.

        :param unary_predicates: the list of unary predicates names
        :param binary_predicates: the list of binary predicates names
        :param unary_clauses: a list of unary clauses. Each clause is a string on the form:
        clause_weight:clause

        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).

        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.

        An example:
           _:nDog,Animal

        :param binary_clauses: a list of binary clauses
        :param activation: activation function
        :param initial_clause_weight: initial value for the cluase weight (if clause is not hard)
        """

        super(RelationalKENN, self).__init__(**kwargs)

        self.unary_predicates = unary_predicates
        self.n_unary = len(unary_predicates)
        self.unary_clauses = unary_clauses
        self.binary_predicates = binary_predicates
        self.binary_clauses = binary_clauses
        self.activation = activation
        self.initial_clause_weight = initial_clause_weight

        self.unary_ke = None
        self.binary_ke = None
        self.join = None
        self.group_by = None

    def build(self, input_shape):
        if len(self.unary_clauses) != 0:
            self.unary_ke = KnowledgeEnhancer(self.unary_predicates, self.unary_clauses, initial_clause_weight=self.initial_clause_weight)
        if len(self.binary_clauses) != 0:
            self.binary_ke = KnowledgeEnhancer(self.binary_predicates, self.binary_clauses, initial_clause_weight=self.initial_clause_weight)

            self.join = Join()
            self.group_by = GroupBy(self.n_unary)

        super(RelationalKENN, self).build(input_shape)

    def call(self, unary, binary, index1, index2, **kwargs):
        """Forward step of Kenn model for relational data.

        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param index1: a vector containing the indices of the first object
        of the pair referred by binary tensor
        :param index1: a vector containing the indices of the second object
        of the pair referred by binary tensor
        """

        if len(self.unary_clauses) != 0:
            u = unary + self.unary_ke(unary)
        else:
            u = unary

        if len(self.binary_clauses) != 0:
            joined_matrix = self.join(u, binary, index1, index2)
            deltas = self.binary_ke(joined_matrix)

            up, bp = self.group_by(u, binary, deltas, index1, index2)
        else:
            up = u
            bp = binary

        return self.activation(up), self.activation(bp)

    def get_config(self):
        config = super(RelationalKENN, self).get_config()
        config.update({'unary_predicates':self.unary_predicates})
        config.update({'unary_clauses': self.unary_clauses})
        config.update({'binary_predicates': self.binary_predicates})
        config.update({'binary_clauses': self.binary_clauses})
        config.update({'activation': self.activation})
        config.update({'initial_clause_weight': self.initial_clause_weight})

        return config