import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from keras import models,layers


def make_float(names, arr):

    lines = open(names, 'r').readlines()
    lst = []
    for _ in range(26):
        lst.append("continuous")
    i = 0
    for line in lines:
        data = line.split(':')[1]
        if not 'continuous' in data:
            data = data.strip()
            data = data.split(',')
            lst[i] = data
        i += 1
    float_arr = []
    for data, info in zip(arr, lst):
        if info is "continuous":
            data = float(data)
        else:
            step = float(1/len(info))
            for i in range(0, len(info)):
                if info[i] == data:
                    break
            data = i * step

        float_arr.append(data)

    return float_arr[2:]


def preprocess(data, names):
    
    lines = open(data, 'r').readlines()
    lines = [line.strip() for line in lines]
    
    
    data = []
    target = []
    
    for line in lines:
        if '?' in line:
            continue
        segment = line.split(',')
        target.append(segment[1])
        data.append(make_float(names, segment))
        
    return preprocessing.normalize(data), target
    
def main():
    
    data, target = preprocess('imports-85-data.csv', 'feature.txt')
    x_data, x_test, y_data, y_test = train_test_split(data, target, test_size = 0.1)

    #model
    model = keras.Sequential()
    model.add(layers.Dense(30, activation = "relu", input_shape = [24]))
    model.add(layers.Dense(15, activation = "relu"))
    model.add(layers.Dense(1))
    
    model.compile(optimizer = keras.optimizers.SGD(0.001), loss = 'mse', metrics = ['mse'])
    
    splitter = KFold(15, shuffle = False)

    y_data = np.array(y_data)
    for train, test in splitter.split(x_data):
        x_train, x_test = x_data[train], x_data[test]
        y_train, y_test = y_data[train], y_data[test]
        model.fit(x_train, y_train, epochs = 200)
        prediction = model.predict(x_test)

    prediction = model.predict(x_test)
    print("=====================\ntest result\n=====================\n")
    for p, t in zip(prediction, y_test):
        print("prediction {} target {} ".format(p[0], t))
    
if __name__ == '__main__':
    main()
