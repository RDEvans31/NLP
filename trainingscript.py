from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd
import numpy as np

import nltk

from flair.data import Corpus
from flair.datasets import ColumnCorpus

from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings
from typing import List
from flair.trainers import ModelTrainer


columns = {0: 'text', 1: 'ner'}

corpus : Corpus = ColumnCorpus('normalisation-project/custom_model_3', columns,
                                train_file='train.txt',
                                test_file='test.txt',
                                dev_file='dev.txt')

tag_type = 'ner'
# make tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings

flair_embedding_forward = FlairEmbeddings('mix-forward')
flair_embedding_backward = FlairEmbeddings('mix-backward')
transformer = TransformerWordEmbeddings('distilbert-base-uncased')

stacked_embeddings = StackedEmbeddings([
                                        transformer
                                       ])

tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=stacked_embeddings,
                                       tag_dictionary=tag_dictionary,
                                       tag_type=tag_type,
                                       use_crf=True)

trainer : ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('normalisation-project/models/resources_v7/taggers/ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              monitor_test=True)