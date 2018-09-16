import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    num_points = len(series) - window_size
    for i in range(num_points):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    
    #print(X)
    #print(y)    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units=5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    letters = ['a','b','c','d','e','f','g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 
               'v','w','x','y','z']
    
    remove_chars = []
    
    for char in text:
        if (char not in punctuation and char not in letters):
            remove_chars.append(char)
    
    for char in remove_chars:
        text = text.replace(char,' ')
   
    return text



### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    idx = 0
    
    while (idx + window_size) < len(text):
        inputs.append(text[idx:idx+window_size])
        outputs.append(text[idx+window_size])
        idx += step_size
    
    return inputs,outputs



# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(units=200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model
