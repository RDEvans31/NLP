import nltk
from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
from dateutil.rrule import *
from dateutil.parser import *
from datetime import *
import inflect

p = inflect.engine()
import os

directory = os.getcwd()

print(directory)


tokenizer = nltk.RegexpTokenizer(r"[\w'-]+")

# for some reason the current working directory accoding to python is not normalisation-project bu loop-webapps

# this defines the best model for
best_model_ner = SequenceTagger.load('resources_v4/taggers/ner/best-model.pt') # FLAIR embedding model

# this defines our trained model for STS
try:
    model_sts = SentenceTransformer('normalisation-project/sts-model')
except:
    model_sts = SentenceTransformer('sts-model')

# this defines the dictionary for medical abbreviations
try:
    f = open('normalisation-project/data/abbreviations.txt', "r")
except:
    f = open('data/abbreviations.txt', "r")

tokenizer = nltk.RegexpTokenizer(r"[\w'-]+")

# this fetches results to evaulate which model to use based on latest results

# this defines the model for NER

best_model_ner = SequenceTagger.load('app/models/best-model.pt') # FLAIR embedding model

# this defines our standard string mappping to RRule

rrule_dict ={
    "once weekly":"FREQ=WEEKLY;BYDAY=MO",
    "twice weekly":"FREQ=WEEKLY;BYDAY=TU,FR",
    "three times weekly":"FREQ=WEEKLY;BYDAY=TU,TH,SU",
    "every other day": "FREQ=DAILY;INTERVAL=2;BYHOUR=9",
    "once daily":"FREQ=DAILY;BYHOUR=9",
    "twice daily":"FREQ=DAILY;BYHOUR=8,20",
    "three times daily":"FREQ=DAILY;BYHOUR=8,13,18",
    "four times daily":"FREQ=DAILY;BYHOUR=8,12,16,20",
    "five times daily":"FREQ=DAILY;BYHOUR=8,11,14,17,20",
    "every morning":"FREQ=DAILY;BYHOUR=9",
    "every night":"FREQ=DAILY;BYHOUR=21",
}

rrules_df = pd.DataFrame.from_dict(rrule_dict, orient='index', columns=['rrule']).reset_index()
rrules_df.columns = ['text', 'rrule']
rrules_df.loc[:, 'embeddings'] = rrules_df.loc[:, 'text'].apply(model_sts.encode)

# this defines the dictionary for medical abbreviations
f = open('app/data/abbreviations.txt', "r")

abb_dict = {}
multiple_abbreviations = []

def configure_abreviations(line):
    strings = line.split(':')
    long_form = ' '.join(strings[1].split())
    if len(long_form.split('/'))>1:
        print(long_form.split('/'))
    abbreviation_list=strings[0].split(',')
    if len(abbreviation_list) > 1:
        multiple_abbreviations.append(abbreviation_list)
    abb = abbreviation_list[0] if len(abbreviation_list) == 1 else abbreviation_list[1]
    abbreviation, text = ' '.join(abb.split()), long_form
    abb_dict[abbreviation] = text

[configure_abreviations(line) for line in f]
multiple_abb_mapping = [(','.join(abb),abb[1]) for abb in multiple_abbreviations]

abb_mappings_dict = {}
for mapping in multiple_abb_mapping:
    temp = list(map(lambda x : x.split(),mapping[0].split(',')))
    key = [item for sublist in temp for item in sublist]
    abb_mappings_dict[tuple(key)] = ''.join(mapping[1].split())

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

def post_process(sentence: str):
    sentence = Sentence(sentence)
    new_string = []
    for i in range(len(sentence)):
        token = sentence[i].text
        try:
            number = p.number_to_words(int(token))
            new_string.append(str(number))
        except:
            new_string.append(token)
    return ' '.join(new_string)

def label(text, model):
    # run NER over sentence
    text=Sentence(process(text))
    model.predict(text)
    ents = [(entity['text'], entity['labels'][0].value) for entity in text.to_dict('ner')['entities']]
    return ents

def find_match(string: str, model):
    embeddings_stan  = rrules_df.loc[:, 'embeddings'].to_list()
    difference =  cosine_similarity([model.encode(string)], embeddings_stan)
    index = difference[0].argmax()
    return index, rrules_df.iloc[index].loc['text'], difference[0][index]

def current_time_string():
    datetime_string = datetime.now().__str__()
    date_string = datetime_string.split()[0].split('-')
    time_string = datetime_string.split()[1].split(':')
    time_string[-1] = '00'
    time_string[-2] = '00'
    return ''.join(date_string+["T"]+time_string)

def take_times(rrule_string: str, start: str, occurences: int = 7):
    # can take set amount prescribed as occurences and schedule will stop when presciption
    count = ";COUNT=" + str(occurences)
    return list(rrulestr( rrule_string+count, dtstart=parse(start) ))

def schedule(string: str, start: str = current_time_string()):
    labels = label(process(string), best_model_ner)
    frequencies = [post_process(item[0]) for item in labels if item[1] == 'FREQUENCY']
    try:
        frequency = frequencies[0]
        processed_frequency = post_process(frequency)
        match_index, match, similarity = find_match(processed_frequency, model_sts)
        return list(map(lambda x: x.__str__(), take_times(rrules_df.iloc[match_index].loc['rrule'], start)))
    except ParserError:
        return 'Invalid start time: '+ start
    except IndexError: # probably means that there are no frequencies
        return 'No frequencies found.'
    except:
        return 'Error: invalid input'

def schedule_datetimes(string: str, start: str = current_time_string()):
    labels = label(process(string), best_model_ner)
    frequencies = [post_process(item[0]) for item in labels if item[1] == 'FREQUENCY']
    try:   
        frequency = frequencies[0]
        processed_frequency = post_process(frequency)
        match_index, match, similarity = find_match(processed_frequency, model_sts)
        return take_times(rrules_df.iloc[match_index].loc['rrule'], start, 50)
    except ParserError:
        return 'Invalid start time: '+ start
    except IndexError: # probably means that there are no frequencies
        return 'No frequencies found.'
    except:
        return frequencies



