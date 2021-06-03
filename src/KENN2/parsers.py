from KENN2.layers.residual.KnowledgeEnhancer import KnowledgeEnhancer
from KENN2.layers.Kenn import Kenn
from KENN2.layers.RelationalKENN import RelationalKENN


def unary_parser(knowledge_file, activation=lambda x: x, initial_clause_weight=0.5, save_training_data=False, **kwargs):
    """
    Takes in input the knowledge file containing only unary clauses and returns a Kenn Layer, 
    with input the predicates and clauses found in the knowledge file.

    :param knowledge_file: path of the prior knowledge file
    """
    with open(knowledge_file, 'r') as kb_file:
        predicates_string = kb_file.readline()
        kb_file.readline()
        clauses = kb_file.readlines()

    predicates = predicates_string[:-1].split(',')

    return Kenn(predicates, clauses, activation, initial_clause_weight, save_training_data, **kwargs)


def unary_parser_ke(knowledge_file, initial_clause_weight=0.5, **kwargs):
    """
    Takes in input the knowledge file containing only unary clauses and returns a Knowledge Enhancer layer, 
    with input the predicates and clauses found in the knowledge file.

    :param knowledge_file: path of the prior knowledge file;
    """
    with open(knowledge_file, 'r') as kb_file:
        predicates_string = kb_file.readline()
        kb_file.readline()
        clauses = kb_file.readlines()

    predicates = predicates_string[:-1].split(',')

    return KnowledgeEnhancer(predicates, clauses, initial_clause_weight, **kwargs)


def relational_parser(knowledge_file, activation=lambda x: x, initial_clause_weight=0.5, **kwargs):
    """
    Takes in input the knowledge file containing both unary and binary clauses and returns a RelationalKENN 
    Layer, with input the predicates and clauses found in the knowledge file.

    :param knowledge_file: path of the prior knowledge file;
    """
    with open(knowledge_file, 'r') as kb_file:
        unary_literals_string = kb_file.readline()
        binary_literals_string = kb_file.readline()

        kb_file.readline()
        clauses = kb_file.readlines()

    u_groundings = [u + '(x)' for u in unary_literals_string[:-1].split(',')]
    b_groundings = [u + '(x)' for u in unary_literals_string[:-1].split(',')] + \
                   [u + '(y)' for u in unary_literals_string[:-1].split(',')] + \
                   [b + '(x.y)' for b in binary_literals_string[:-1].split(',')] + \
                   [b + '(y.x)' for b in binary_literals_string[:-1].split(',')]

    unary_clauses = []
    binary_clauses = []

    reading_unary = True
    for clause in clauses:
        if clause[0] == '>':
            reading_unary = False
            continue

        if reading_unary:
            unary_clauses.append(clause)
        else:
            binary_clauses.append(clause)

    return RelationalKENN(
        u_groundings,
        b_groundings,
        unary_clauses,
        binary_clauses,
        activation,
        initial_clause_weight)
