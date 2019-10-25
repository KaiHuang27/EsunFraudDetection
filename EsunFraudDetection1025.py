#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esun credit card fraud detection
Created on Mon Sep 16 13:42:28 2019
@author: kai
"""

import numpy as np
from numpy.random import permutation
from numpy import array_split, concatenate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import itertools
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score 


class FraudDataPreprocessing(object):
    '''
    Training data preprocessing
    '''
    
    def __init__(self, data_file, fill_na=True):
        '''
        Loads data file and prepares data
        :param data_file: csv file name
        :param fill_na: fill nan value, or drop nan value automatically
        '''
        self.raw_data = pd.read_csv(data_file)
        self.train_data = self.preprocessing(self.raw_data, fill_na)
        
    def preprocessing(self, data, fill_na=True):
        '''
        :param data: dataFrame of raw data
        :return: dataFrame of preprocessed data
        '''
        if fill_na == True:
            data = data.fillna(0)
        else:
            data = data.dropna().reset_index()

        # feature engineering      
        # turn y/n to 1/0
        yn_mapping = {'N':0, 'Y':1}
        map_lst = ['flbmk', 'ecfg', 'flg_3dsmk', 'insfg', 'ovrlt']
        for k in map_lst:
            data[k] = data[k].map(yn_mapping)
        
        # new feature 'stocn_foreign'
        data['stocn_foreign'] = 0
        data.loc[data['stocn']!=102,['stocn_foreign']] =1
        
        # loctm 取出 時分秒
        second = data['loctm'] % 100
        minute = (data['loctm'] % 10000 - second) // 100
        hour = data['loctm']//10000
        data['time'] = hour * 3600 + minute * 60 + second
        data['time_from_base_date'] = data['time'] + ((data['locdt'] - 1) * 86400)
            
        # 'iterm' 分箱 0為2 1,2為1 3~8為0
        iterm_map = {0:2, 1:1, 2:1, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
        data['iterm_map'] = data['iterm'].map(iterm_map)
        
        _, canos = pd.factorize(data['cano'])
        #信用卡當日使用次數/平均日使用次數
        data['all_NoT'] = 0
        data['today_NoT'] = 0 #today's number of transaction
        data['avg_NoT'] = 0
        
        for cano in canos:
            mask = (data['cano'] == cano)

            #是否為重複刷卡
            for i in data[mask].index:
                conam = data.ix[i, 'conam']
                absolute_time = data.ix[i, 'time_from_base_date']
                
                for j in data[mask].index:
                    if i < j:
                        if abs(conam - data.ix[j, 'conam']) < conam * 0.01:
                            if abs(absolute_time - data.ix[j, 'time_from_base_date']) < 600:
                                data.ix[i, 'duplicate_transaction'] = 1
            #信用卡當日使用次數/平均日使用次數
            _, dts = pd.factorize(data[mask]['locdt'])    
            data.loc[mask, 'all_NoT'] = len(data[mask])
            data.loc[mask, 'avg_NoT'] = len(data[mask]) / len(dts)
            for dt in dts:
                data.loc[mask & (data['locdt'] == dt),'today_NoT'] = len(data[mask & (data['locdt'] == dt)])
                
        data['dev_NoT'] = math.exp(data['today_NoT'] / data['avg_NoT'])

        # get bank acount number
        _, bacno = pd.factorize(data['bacno'])
        
        # create new features from each bacno
        data['card_amnt'] = np.nan
        data['time_distance_avg'] = 0
        data['time_distance_std'] = 0
        data['normalized_conam'] = 0
        data['duplicate_transaction'] = 0

        for no in bacno:
            mask = (data['bacno'] == no)
                                    
            # 計算與其他筆交易的時間距離平均與標準差
            times = data[mask]['time']
            for t_index1 in times.index:
                time_distance = []
                for t_index2 in times.index:
                    if t_index1 < t_index2:
                        if times[t_index1] > times[t_index2]:
                            time_distance.append(min(times[t_index1] - times[t_index2], times[t_index2] + 86400 - times[t_index1]))
                        else:
                            time_distance.append(min(times[t_index2] - times[t_index1], times[t_index1] + 86400 - times[t_index2]))                  
                    else: 
                        pass     
                data.ix[t_index1, 'time_distance_avg'] = np.mean(time_distance)
                data.ix[t_index1, 'time_distance_std'] = np.std(time_distance)
                
            # amount of bank card for each bank acount
            cardno = data[mask]['cano']
            _, card_lst = pd.factorize(cardno)
            card_amnt = len(card_lst)
            data.loc[mask, 'card_amnt'] = card_amnt
            
            # normalized conam
            conam_std = np.std(data[mask]['conam'])
            conam_mean = np.mean(data[mask]['conam'])
            data.loc[mask, 'normalized_conam'] = (data[mask]['conam'] - conam_mean) / conam_std 
        
        # 將分類變數轉為平均盜刷率
        data['new_score'] = 0
        grouped_fraud = data['fraud_ind'].groupby([data['contp'], data['etymd'],data['iterm'], data['stscd'], data['hcefg']]).mean()
        idxs = tuple(grouped_fraud.index)
        for idx in idxs:
            mask = (data['contp'] == idx[0]) & (data['etymd'] == idx[1]) & (data['iterm'] == idx[2]) & (data['stscd'] == idx[3]) & (data['hcefg'] == idx[4])
            data.loc[mask, ['new_score']] = grouped_fraud[idx]

        return data
        
        
class FraudDetection(object):
    '''
    credit card fraud detection
    '''
    
    def __init__(self, data_file, sample_method=None):
        '''
        Loads data file 
        :param data_file: dataframe of data
        '''
        self.data_frame = data_file
        self.features = self.data_frame.columns[self.data_frame.columns != 'fraud_ind']
        self.train_data = self.dataSampling(self.data_frame, method=sample_method)
    
    def dataSampling(self, data, method):
        '''
        :param data: pd.DataFrame
        :param method: method of unbalanced data sampling
        :return : dataframe of sampled data
        '''
        if method == None :    
            return self.data_frame
        
        else:
            sampler = self.getSampler(method)
            data_x ,data_y = sampler.fit_sample(data[self.features], self.get_label(data))
            data_x = pd.DataFrame(data_x, columns=self.features)
            data_y = pd.DataFrame(data_y, columns=['fraud_ind'])
            sampled_data = data_x
            sampled_data['fraud_ind'] = data_y['fraud_ind']
            return sampled_data
    
    
    def getSampler(self, method):
        '''
        :param: method to process imbalanced data(RUS, ROS, SMOTE, ADASYN)
        :return: sampler
        '''
        if method == 'SMOTE':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=27)
            return sampler
        
        elif method == 'ROS':
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=27)
            return sampler
        
        elif method == 'ADASYN':
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=27)
            return sampler
            
        elif method == 'RUS':
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=27)
            return sampler
            
        else:
            return self.data_frame
            print('Data sampling was unsuccessfully')

    @staticmethod    
    def get_label(data):
        y = data['fraud_ind']
        return y 
            
    def validation_data(self, folds):
        '''
        Performs data splitting, classifier training and prediction for given #folds
        :param folds: number of folds
        :return : list of numpy.array pairs (prediciton, expected)
        '''
        data = self.train_data
        response = []
        
        assert len(data) > folds
        
        perms = array_split(permutation(len(data)), folds)
        
        for i in range(folds):
            train_idxs = list(range(folds))
            train_idxs.pop(i)
            train = []
            
            for idx in train_idxs:
                train.append(perms[idx])
            
            train  = concatenate(train)
            
            test_idx = perms[i]
            
            training = data.iloc[train]
            test_data = data.iloc[test_idx]
            
            y = self.get_label(training)
            classifier = self.train(training[self.features], y)
            predictions = classifier.predict(test_data[self.features])
            
            expected = self.get_label(test_data)
            response.append([predictions, expected])
        
        return response
    

class FraudClassifier(FraudDetection):
    '''
    Evaluate classifier
    '''
    def validate(self, folds):
        '''
        Evaluate classifier using confusion matrices
        :param fold: number of folds
        :return: list of confusion metrices per fold
        '''
        result = []
        
        for pred, actual in self.validation_data(folds):
            confusion_matrices = self.confusion_matrix(actual, pred)
            acc = accuracy_score(actual, pred)
            f1 = f1_score(actual, pred)
            result.append([confusion_matrices, acc, f1])
        
        return result
        
    @staticmethod
    def confusion_matrix(test, train):
        return pd.crosstab(test, train, rownames=['actual'], colnames=['preds'])
    
    def fraudPredict(self, test_data):
        '''
        :prarm test_file: DataFrame 
        '''
        y = self.get_label(self.train_data)
        classifier = self.train(self.train_data[self.features], y)
        predictions = classifier.predict(test_data[self.features])
        
        return predictions


class FraudXGB(FraudClassifier):
    '''
    Implementation of fraud detection with XGBclassifier
    '''   
    def train(self, X, Y):
        '''
        Train classifier
        :param X: training input samples
        :param Y: target values
        :return: classifier
        '''
        
        classifier = XGBClassifier(max_depth=20, n_estimators=200)
        classifier = classifier.fit(X, Y)
        
        return classifier
        
        
if __name__ == '__main__':
    trainData = FraudDataPreprocessing('sample50k.csv').getPreprocessedData()
    testData = FraudDataPreprocessing('test.csv').getPreprocessedData()
    
    xgb = FraudXGB(trainData)
    validation = xgb.validate(5)
    yhat = xgb.fraudPredict(testData)
    
    submit = testData['txkey']
    submit['fraud_ind'] = yhat
    submit.to_csv('submit.csv')



        
           
        
        
        
        
        
        
        
        
            
