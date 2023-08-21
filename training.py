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

### tokenizing, stemming, bag of words

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


### TRAINING

all_words = []
tag_list = []
pattern_tag = []
ignore_tokens = [",", ".", "!", "?"]

for intent in intents_data["intents"]:
    tag = intent["tag"]
    tag_list.append(tag)
    for pattern in intent["patterns"]:
        tokenized_pattern = tokenize(pattern)
        all_words.extend(tokenized_pattern)
        pattern_tag.append((tokenized_pattern, tag))

all_words = [stem(w) for w in all_words if w not in ignore_tokens]
all_words = sorted(set(all_words))
tag_list = sorted(set(tag_list))

pattern_train = []
tag_train = []

for (pattern, tag) in pattern_tag:
    bag = bag_of_words(pattern, all_words)
    pattern_train.append(bag)

    label = tag_list.index(tag)
    tag_train.append(label)

pattern_train = np.array(pattern_train)
tag_train = np.array(tag_train)

### training parameters 

num_epochs = 1000
batch_size = 8
learning_rate = .001
input_size = len(all_words)
hidden_size = 8
output_size = len(tag_list)

class ChatDataset(Dataset):
    def __init__(self):
        self.num_samples = len(pattern_train)
        self.x_data = pattern_train
        self.y_data = tag_train

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]

    def __len__(self):
        return self.num_samples
    
chat_data = ChatDataset()
train_load = DataLoader(dataset=chat_data, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size=input_size, hidden_size=batch_size, output_size=output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_load:
        words = words.to(device, dtype=torch.float) 
        labels = labels.to(device, dtype=torch.long)
        output = model(words)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

model_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tag_list": tag_list
}

FILE = "model_data.pth"
torch.save(model_data, FILE)