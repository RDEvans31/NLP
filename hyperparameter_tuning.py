from flair.data import Sentence
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter, SequenceTaggerParamSelector
from flair.data import Corpus
from flair.datasets import ColumnCorpus
import os
import nltk

from abb_mappings_dict import abb_mappings_dict, abb_dict

directory = os.getcwd()

print(directory)

tokenizer = nltk.RegexpTokenizer(r"[\w'-]+")

columns = {0: 'text', 1: 'ner'}


corpus : Corpus = ColumnCorpus('custom_model_3', columns,
                                train_file='train.txt',
                                test_file='test.txt',
                                dev_file='dev.txt')

def process(text):
    tokens = list(map(lambda x: x.casefold(), tokenizer.tokenize(text)))
    keys1 = abb_mappings_dict.keys() # multiple abbreviations
    for i in range(len(tokens)):
        token = tokens[i]
        mult_abb = next((x for x in keys1 if token in x), None)
        if mult_abb != None:
            token = abb_mappings_dict[mult_abb]
        
        if token in abb_dict.keys():
            tokens[i] = abb_dict[token]
    
    return ' '.join(tokens)

def label(text, model):
    # run NER over sentence
    text=Sentence(process(text))
    model.predict(text)
    ents = [(entity.text, entity.get_label("ner").value) for entity in text.get_spans('ner')] #getting NER spans
    return ents

glove_embedding = WordEmbeddings('glove')

flair_embedding_forward = FlairEmbeddings('mix-forward')
flair_embedding_backward = FlairEmbeddings('mix-backward')

# define your search space
search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
    StackedEmbeddings( [glove_embedding, flair_embedding_forward, flair_embedding_backward]),
])
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[128, 256])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])

param_selector = SequenceTaggerParamSelector(corpus,
                                             'ner',
                                             'results/',
                                             training_runs=3,
                                             max_epochs=50
)

# start the optimization
param_selector.optimize(search_space, max_evals=100)