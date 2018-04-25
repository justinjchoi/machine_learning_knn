## Justin Choi (jc8mc)
## November 27th 2017 
## CS 4710: Artificial Intelligence

import numpy as np 
import pandas as pd
import random
from random import randrange
import collections
import copy

def split_data(user_dataset):
    fold_size = int(len(user_dataset) / 6)
    copy = list(user_dataset)
    dataset_split = list()
    i = 0
    while i < 6:
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(copy))
            fold.append(copy.pop(index))
        dataset_split.append(fold)
        i += 1
    return dataset_split

def knn(user_data, predict, k):
    distance_list = []    
    for each_group in user_data:
        for features in user_data[each_group]:
            my_dist = np.linalg.norm(np.array(features) - np.array(predict))
            distance_list.append([my_dist, each_group])

    sub_result = [i[1] for i in sorted(distance_list)[:k]]
    result = collections.Counter(sub_result).most_common(1)[0][0]
    return result

def main(): 
    my_dataframe = pd.DataFrame()
    with open("training.csv", 'r') as f:
        for each_line in f:
            my_dataframe = pd.concat( [my_dataframe, pd.DataFrame([tuple(each_line.strip().split(','))])], ignore_index=True )
    
    my_dataframe.drop(0, 1, inplace = True)
    
    list_data = my_dataframe.values.tolist()
    
    count = 0
    for row in list_data:
        modified = []
        for data in row:
            if type(data) is str:
                modified.append(data[1:-1])
        list_data[count] = modified
        count += 1
    
    read_ingredients = pd.read_csv("ingredients.txt", header=None)
    read_ingredients = read_ingredients.values.tolist()
        
    num_rows = 0
    for row in list_data:
        cuisine = row[0]
        this_row = copy.deepcopy(read_ingredients)
        datacount = 0
        for ingredient in this_row:
            if ingredient[0] in row:
                this_row[datacount] = 1
                datacount += 1
            else:
                this_row[datacount] = 0
                datacount += 1
        list_data[num_rows] = this_row
        cuisine_to_num = 0 
        if cuisine == "brazilian":
            cuisine_to_num = 0
        elif cuisine == "british": 
            cuisine_to_num = 1
        elif cuisine == "cajun_creole":
            cuisine_to_num = 2
        elif cuisine == "chinese":
            cuisine_to_num = 3
        elif cuisine == "filipino":
            cuisine_to_num = 4
        elif cuisine == "french":
            cuisine_to_num = 5
        elif cuisine == "greek":
            cuisine_to_num = 6
        elif cuisine == "indian":
            cuisine_to_num = 7
        elif cuisine == "irish":
            cuisine_to_num = 8
        elif cuisine == "italian":
            cuisine_to_num = 9
        elif cuisine == "jamaican":
            cuisine_to_num = 10
        elif cuisine == "japanese":
            cuisine_to_num = 11
        elif cuisine == "korean":
            cuisine_to_num = 12
        elif cuisine == "mexican":
            cuisine_to_num = 13
        elif cuisine == "moroccan":
            cuisine_to_num = 14
        elif cuisine == "russian":
            cuisine_to_num = 15
        elif cuisine == "southern_us":
            cuisine_to_num = 16
        elif cuisine == "spanish":
            cuisine_to_num = 17
        elif cuisine == "thai":
            cuisine_to_num = 18
        elif cuisine == "vietnamese":
            cuisine_to_num = 19
        else:
            cuisine_to_num = -1
        list_data[num_rows].append(cuisine_to_num)
        num_rows += 1
    
    
    accuracy_list = []
    error_list = []
    random.shuffle(list_data)
    folds = split_data(list_data)
    a = 0 
    while a < 6:
        train_set = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[]}
        test_set = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[]}
        
        train_data = []
        test_data = folds[a]
        
        for i in range(6):
            if i != a: 
                train_data += folds[a]
    
        for i in train_data:
            train_set[i[-1]].append(i[:-1])
        
        for i in test_data:
            test_set[i[-1]].append(i[:-1])
    
        correct = 0
        total = 0
    
        for each_group in test_set:
            for data in test_set[each_group]:
                vote = knn(train_set, data, k=35)
                if each_group == vote:
                    correct += 1
                total += 1
        
        accuracy = correct/total
        error = 1/len(test_data) * (total-correct)
        accuracy_list.append(accuracy)
        error_list.append(error)
        print("Fold ", a+1, ": ")
        print('Accuracy: ', accuracy)
        print('Error: ', error)
        a += 1
    
    avg_accuracy = np.mean(accuracy_list)
    avg_error = np.mean(error_list)
    print("Average Accuracy:", avg_accuracy)
    print("Average Error:", avg_error)
    
if __name__ == "__main__": 
    main()
