# this defines the dictionary for medical abbreviations
try:
    f = open('normalisation-project/data/abbreviations.txt', "r")
except:
    f = open('data/abbreviations.txt', "r")

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
