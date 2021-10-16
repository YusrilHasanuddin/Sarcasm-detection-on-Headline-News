import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


def sarcasm_detection():
    #Get the dataset
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    #Initializazing variables
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    #Later this variable use to get the sentence and its labels
    sentences = []
    labels = []
    
    #Regex for retrieve right sentence
    with open("sarcasm.json", 'r') as f:
        datalib = json.loads("[" + f.read().replace("}\n{", "},\n{") + "]")

    #Append the data
    for item in datalib:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    #Splitting dataset
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    #Add tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    #To decode question mark
    def decode_sentence(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    #This code to adding sequences and padding for getting the sentence, here I use the max length of a data.
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    #The layer. from last iteration I use LSTM, but it don't make the accuracy increase.
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    #For compile and train the data using binary_crossentropy because the output is binary (is_sarcastic [0 or 1])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    process = model.fit(training_padded, training_labels, epochs=10, validation_data=(testing_padded, testing_labels), verbose=1)

    #Plotting the model
    acc = process.history['accuracy']
    val_acc = process.history['val_accuracy']
    loss = process.history['loss']
    val_loss = process.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()

    plt.show()
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = sarcasm_detection()
    model.save("model-sarcasm-detection.h5")
