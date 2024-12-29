import random
import nltk
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from tf_keras.models import Sequential, load_model
from tf_keras.layers import Dense, Dropout
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import json
import pickle

stemmer = LancasterStemmer()

with open('data.json') as file:
    json_data = json.load(file)

with open("data.pickle", 'wb') as pickle_file:
    pickle.dump(json_data,pickle_file)

with open("data.pickle", "rb") as binary_file:
    pickle_data = pickle.load(binary_file)

tags = []
pattern_words = []
tokenized_patterns = []
pattern_tags = []

for intent in pickle_data["intents"]:
    for pattern in intent["patterns"]:
        tokenized_words = word_tokenize(pattern)
        pattern_words.append(tokenized_words)
        tokenized_patterns.append(tokenized_words)
        pattern_tags.append(intent["tag"])

        if intent["tag"] not in tags:
            tags.append(intent["tag"])


stemmed_words = [[stemmer.stem(word) for word in inner_lists] for inner_lists in pattern_words]
flattened_words = [word for sublist in stemmed_words for word in sublist]
unique_sorted_words = sorted(set(flattened_words))

sorted_tags = sorted(tags)

training = []
output = []

output_empty = [0] * len(tags)


for idx, pattern in enumerate(tokenized_patterns):
    stemmed_pattern_words = [stemmer.stem(word.lower()) for word in pattern]
    bag = [1 if word in stemmed_pattern_words else 0 for word in unique_sorted_words]
    # Output is a '0' for each tag and '1' for the current tag
    output_row = list(output_empty)
    output_row[tags.index(pattern_tags[idx])] = 1

    training.append(bag)
    output.append(output_row)
# Convert to numpy arrays
training = np.array(training)
output = np.array(output)


# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(training[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output[0]), activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model.fit(training, output, epochs=200, batch_size=5, verbose=1)



# Save the model
model.save('chatbot_model.keras')

# Save the data structures
with open('words.pkl', 'wb') as f:
    pickle.dump(unique_sorted_words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(tags, f)


# Load the model and data structures
model = load_model('chatbot_model.keras')
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        print(return_list)
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, pickle_data)
    return res



print("Start talking with the bot (type 'quit' to stop)!")
while True:
    message = input("")
    if message.lower() == "quit":
        break
    response = chatbot_response(message)
    print(response)