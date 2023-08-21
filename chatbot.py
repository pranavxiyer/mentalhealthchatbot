import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import json
from model import NeuralNet

### load training data, generate response

stemmer = PorterStemmer()

with open('intents.json') as json_file:
    intents_data = json.load(json_file)

def tokenize(input):
    return nltk.word_tokenize(input)

def stem(word):
    lowercase = word.lower()
    return stemmer.stem(lowercase)

def bag_of_words(tokenized_sentence, all_words):
    word_count = [0.0 for i in range(len(all_words))]
    stemmed = [stem(word) for word in tokenized_sentence]
    for i, w in enumerate(all_words):
        if w in stemmed:
            word_count[i] = 1.0
    return word_count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE = "model_data.pth"
model_data = torch.load(FILE)

input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data["all_words"]
tag_list = model_data["tag_list"]
model_state = model_data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

print("I am a psychotherapist chatbot, talk to me about your emotions! Type 'quit' to exit.")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break
    tokenized_sentence = tokenize(sentence)
    X = bag_of_words(tokenized_sentence, all_words)
    X = np.array(X)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device, dtype=torch.float)

    output = model(X)
    temp, predicted = torch.max(output, dim=1)
    tag = tag_list[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    tag_prob = probs[0][predicted.item()]

    if tag_prob.item() >= .75:
        for intent in intents_data["intents"]:
            if tag == intent["tag"]:
                print(random.choice(intent["responses"]))
    else:
        print("Sorry, I do not understand. Could you please elaborate?")