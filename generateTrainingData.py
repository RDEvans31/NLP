import random

quantities = ["one", "two", "three", "four", "five", "1", "2", "3", "4", "5"]
frequencies = ["daily", "twice a day", "three times a day", "once every two days", "every 8 hours"]

def generate_sentence(template, quantity, frequency):
    return template.format(quantity=quantity, frequency=frequency)

def generate_bio_annotations(quantity, frequency, template):
    tokens = template.format(quantity=quantity, frequency=frequency).split()
    labels = []
    for token in tokens:
        if token in quantity.split():
            labels.append("B-QUANTITY" if token == quantity.split()[0] else "I-QUANTITY")
        elif token in frequency.split():
            labels.append("B-FREQUENCY" if token == frequency.split()[0] else "I-FREQUENCY")
        else:
            labels.append("O")
    return list(zip(tokens, labels))

templates = [
    "take {quantity} {frequency}",
    "{quantity} should be taken {frequency}",
    "it's advised to take {quantity} {frequency}",
    "you should take {quantity}, {frequency}",
    "consume {quantity} {frequency}",
]

sentences = []
annotations = []

for template in templates:
    for quantity in quantities:
        for frequency in frequencies:
            sentence = generate_sentence(template, quantity, frequency)
            annotation = generate_bio_annotations(quantity, frequency, template)
            sentences.append(sentence)
            annotations.append(annotation)

for sentence, annotation in zip(sentences, annotations):
    print(sentence)
    print(annotation)
    print()
