import urllib
from urllib.parse import urlparse
import httplib2 as http
import json
import requests
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score

class BusStopFile:

    def __init__(self, file_path, df=None):
        self.df = df
        self.file_path = file_path
        self.uri= 'http://datamall2.mytransport.sg/ltaodataservice/'
        
        # read the api key from a file
        with open('api_key.txt', 'r') as f:
            self.api_key = f.read().strip()
        self.method = 'GET'
        self.body = ''
        self.headers = { 'AccountKey' : self.api_key, 'accept' : 'application/json'}
        self.h = http.Http()

    def get_stop_data(self, start, end):
        self.start = start
        self.end = end
        
        # specify the target web page
        for i in range(self.start,self.end,500): # there are currently only 5000+ bus stops and the lta api limits us by 500 each pull
            
            skip = '?$skip=' + str(i)
            target = urlparse(self.uri + self.file_path + skip)
            print(f'The target url is {target.geturl()}')
            
            response, content = self.h.request(target.geturl(),self.method,self.body,self.headers)
            data = json.loads(content)

            df = pd.DataFrame(data['value'])
            self.df = pd.concat([self.df, df], ignore_index=True)

        return self.df
    
    def get_stop_traffic(self, date=None):
        self.date = date
        target = urlparse(self.uri + 'PV/' + self.file_path + '?Date=' + self.date)
        print(f'The target url is {target.geturl()}')
        
        try:
            response, content = self.h.request(target.geturl(),self.method,self.body,self.headers)
            data = json.loads(content)
            print(data['value'])
        except:
            print(data['fault']['faultstring'])
class Passenger_Data:
    
    def __init__(self,stop_df,mcd_df):
        self.matrix = pd.DataFrame(columns=stop_df['BusStopCode'].unique(), index=range(len(mcd_df)))
        print(f'Empty matrix initialized with shape {self.matrix.shape}')
    
    def traffic_volume(self, distance, mcd_df, stop_df, traffic_df):
        
        # create an empty dataframe with len mcd_df
        num_stops = pd.DataFrame(columns=[f'num_stops{distance}'], index=range(len(mcd_df)))

        # input a value of "1" in the matrix if the distance between the bus stop and mcd is less than the distance specified
        for mcd in range(len(mcd_df)):
            count = 0
            for busstop in range(len(stop_df)):
                dist = geodesic((stop_df['Latitude'][busstop], stop_df['Longitude'][busstop]), \
                                (mcd_df['Latitude'][mcd], mcd_df['Longitude'][mcd])).km
                if dist < distance:
                    self.matrix.iloc[mcd,busstop] = 1
                    count += 1
            num_stops.iloc[mcd,0] = count
            break
        
        # fill remaining values with 0 in order to multiply the values
        self.matrix.fillna(0, inplace=True)
        self.matrix = self.matrix.transpose()
        
        # multiply the traffic values to the matrix
        traffic_in = self.matrix.multiply(traffic_df['TOTAL_TAP_IN_VOLUME'], axis=0)
        traffic_out = self.matrix.multiply(traffic_df['TOTAL_TAP_OUT_VOLUME'], axis=0)
        
        traffic_in = traffic_in.transpose()
        traffic_in = traffic_in.sum(axis=1)
        traffic_in = traffic_in.to_frame()
        traffic_in.columns = [f'tap_in_traffic{distance}']
        
        traffic_out = traffic_out.transpose()
        traffic_out = traffic_out.sum(axis=1)
        traffic_out = traffic_out.to_frame()
        traffic_out.columns = [f'tap_out_traffic{distance}']

        return traffic_in, traffic_out, num_stops

class Distance_Data:
    
    def __init__(self, mcd_df):
        self.matrix = pd.DataFrame(columns=['hawker_count'], index=range(len(mcd_df)))
        # print(f'Empty matrix initialized with shape {self.matrix.shape}')
    
    def num_count(self, distance, mcd_df, hdb_df, col_name):

        # input a value of "1" in the matrix if the distance between the bus stop and mcd is less than the distance specified
        for mcd in range(len(mcd_df)):
            count=0
            for hdb in range(len(hdb_df)):
                dist = geodesic((hdb_df['latitude'][hdb], hdb_df['longitude'][hdb]), \
                                (mcd_df['latitude'][mcd], mcd_df['longitude'][mcd])).km
                if dist < distance:
                    count+=1
            self.matrix.iloc[mcd,0] = count
            self.matrix.columns = [f'{col_name}{distance}']
            
        
        return self.matrix

    def single_num_count(self, distance, latitude, longitude, hdb_df, col_name):

        # input a value of "1" in the matrix if the distance between the bus stop and mcd is less than the distance specified
        count=0
        for hdb in range(len(hdb_df)):
            dist = geodesic((hdb_df['latitude'][hdb], hdb_df['longitude'][hdb]), \
                            (latitude, longitude)).km
            if dist < distance:
                count+=1
        
        return count
    
    def traffic_count(self, mcd_df, stop_df, bus_data, distance_list):
        
        # create an empty numpy array
        to_df_array = np.empty((0,384))

        for mcd in range(len(mcd_df)):
            # create an empty 5082 x 4 numpy array
            traffic_array = np.zeros((len(stop_df), len(distance_list)))
            
            for stop in range(len(stop_df)):
                dist = geodesic((stop_df['latitude'][stop], stop_df['longitude'][stop]), \
                                (mcd_df['latitude'][mcd], mcd_df['longitude'][mcd])).km
                if dist<distance_list[0]:
                    traffic_array[stop,0] = 1
                    if dist<distance_list[1]:
                        traffic_array[stop,1] = 1
                        if dist<distance_list[2]:
                            traffic_array[stop,2] = 1
                            if dist<distance_list[3]:
                                traffic_array[stop,3] = 1
            
            # multiply the 2 arrays to get a 96 x 4 array
            mcd_array = np.dot(bus_data,traffic_array)

            # convert mcd_array to a 1 x 384 array by unstacking the rows
            mcd_array = mcd_array.reshape(1, -1)

            # add it to the empty array
            to_df_array = np.append(to_df_array, mcd_array, axis=0)
            print(f'Number of McDonalds done: {len(to_df_array)}')
        
        return to_df_array
    
    def single_traffic_count(self, lat, long, stop_df, bus_data, distance_list):
        
        # create an empty numpy array
        to_df_array = np.empty((0,384))

        # create an empty 5082 x 4 numpy array
        traffic_array = np.zeros((len(stop_df), len(distance_list)))
        
        for stop in range(len(stop_df)):
            dist = geodesic((stop_df['latitude'][stop], stop_df['longitude'][stop]), \
                            (lat, long)).km
            if dist<distance_list[0]:
                traffic_array[stop,0] = 1
                if dist<distance_list[1]:
                    traffic_array[stop,1] = 1
                    if dist<distance_list[2]:
                        traffic_array[stop,2] = 1
                        if dist<distance_list[3]:
                            traffic_array[stop,3] = 1
        
        # multiply the 2 arrays to get a 96 x 4 array
        mcd_array = np.dot(bus_data,traffic_array)

        # convert mcd_array to a 1 x 384 array by unstacking the rows
        mcd_array = mcd_array.reshape(1, -1)

        # add it to the empty array
        to_df_array = np.append(to_df_array, mcd_array, axis=0)
        print(f'Number of McDonalds done: {len(to_df_array)}')
        
        return to_df_array
            
    def old_traffic_count(self, distance, mcd_df, bus_df, day_type, time_type):
        filtered_df = bus_df[bus_df['TIME_PER_HOUR']==time_type]
        filtered_df = filtered_df[filtered_df['DAY_TYPE'].isin([day_type])]
        filtered_df = filtered_df.groupby(['BusStopCode','latitude','longitude'])[['in','out']].sum()
        filtered_df.reset_index(inplace=True)
        # set index as BusStopCode
        filtered_df.set_index('BusStopCode', inplace=True)

        col = list(filtered_df.index)
        matrix = pd.DataFrame(columns=col, index=range(len(mcd_df)))

        # input a value of "1" in the matrix if the distance between the bus stop and mcd is less than the distance specified
        for mcd in range(len(mcd_df)):
            for stop in range(len(filtered_df)):
                dist = geodesic((filtered_df['latitude'][stop], filtered_df['longitude'][stop]), \
                                (mcd_df['latitude'][mcd], mcd_df['longitude'][mcd])).km
                if dist < distance:
                    matrix.iloc[mcd,stop] = 1

        matrix = matrix.transpose()
        
        # multiply the traffic values to the matrix
        traffic_in = matrix.multiply(filtered_df['in'], axis=0)
        traffic_out = matrix.multiply(filtered_df['out'], axis=0)

        traffic_in = traffic_in.transpose()
        traffic_in = traffic_in.sum(axis=1)
        traffic_in = traffic_in.to_frame()
        traffic_in.columns = [f'in_{day_type}_{time_type}_{distance}']
        
        traffic_out = traffic_out.transpose()
        traffic_out = traffic_out.sum(axis=1)
        traffic_out = traffic_out.to_frame()
        traffic_out.columns = [f'out_{day_type}_{time_type}_{distance}']

        return traffic_in, traffic_out

class Evaluate:

    def __init__(self):
        pass

    def mean_metrics(self, wrong_prediction, metrics, mcd_df_model, evaluate, right_evaluate):
        
        for j in evaluate.index:
            w = []
            r = []
            l = []
            # print out the McDonald's we are evaluating
            title = evaluate.loc[j,:].title
            print(title)

            for i in metrics:
            # print the difference between their values and the average
                # the mean of the whole dataset
                actual_mean = mcd_df_model[i].mean()
                # the mean of the right predictions
                right_mean = right_evaluate[right_evaluate['y_pred']==wrong_prediction][i].mean()
                # the mean of the wrong predictions
                wrong_mean = evaluate.loc[j,i]
                wrong = (actual_mean-wrong_mean)/actual_mean*100
                right = (actual_mean-right_mean)/actual_mean*100
                w.append(wrong)
                r.append(right)
                l.append(i)
                print(f"the % difference between {i} for right predicted values and {title} is {wrong-right:.2f}%")
                # print(f"the % difference between the average of {i} and the wrong prediction of {wrong_prediction} is {(actual_mean-wrong_mean)/actual_mean*100:.2f}%")
                # print(f"the % difference between the average of {i} and the right prediction of {wrong_prediction} is {(actual_mean-right_mean)/actual_mean*100:.2f}%")
            

            # set the figure size to be wider
            plt.figure(figsize=(20,10))
            # Set the width of the bars
            bar_width = 0.35

            # Set the positions of the bars on the x-axis
            r1 = range(len(w))
            r2 = [x + bar_width for x in r1]

            # Create the clustered bar chart
            plt.bar(r1, w, color='red', width=bar_width, edgecolor='grey', label='Wrong Prediction % difference from Mean')
            plt.bar(r2, r, color='green', width=bar_width, edgecolor='grey', label='Right Prediction % difference from Mean')
            
            # rotate xticks by 45 degrees
            plt.xticks([r + bar_width for r in range(len(w))], l, rotation=45)

            # Add a legend
            plt.legend()

            # Show the chart
            plt.show()
        
            

    def median_metrics(self, wrong_prediction, metrics, mcd_df_model, evaluate, right_evaluate):
        w = []
        r = []
        l = []
        for i in metrics:
        # print the median
            # the median of the whole dataset
            actual_median = mcd_df_model[i].median()
            # the median of the right predictions
            right_median = right_evaluate[right_evaluate['y_pred']==wrong_prediction][i].median()
            # the median of the wrong predictions
            wrong_median = evaluate[evaluate['y_pred']==wrong_prediction][i].median()
            if actual_median == 0:
                pass
            else:
                wrong = (actual_median-wrong_median)/actual_median*100
                right = (actual_median-right_median)/actual_median*100
                w.append(wrong)
                r.append(right)
                l.append(i)
                print(f"the % difference between {i} for right and wrong predictions is {wrong-right:.2f}%")
            # print(f"the % difference between the median of {i} and the wrong prediction of {wrong_prediction} is {(actual_median-wrong_median)/actual_median*100:.2f}%")
            # print(f"the % difference between the median of {i} and the right prediction of {wrong_prediction} is {(actual_median-right_median)/actual_median*100:.2f}%")
        
        
        # set the figure size to be wider
        plt.figure(figsize=(20,10))
        # Set the width of the bars
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        r1 = range(len(w))
        r2 = [x + bar_width for x in r1]

        # Create the clustered bar chart
        plt.bar(r1, w, color='red', width=bar_width, edgecolor='grey', label='wrong_prediction_median_difference')
        plt.bar(r2, r, color='green', width=bar_width, edgecolor='grey', label='right_prediction_median_difference')
        
        # rotate xticks by 45 degrees
        plt.xticks([r + bar_width for r in range(len(w))], l, rotation=45)

        # Add a legend
        plt.legend()

        # Show the chart
        plt.show()

class Model:

    def __init__(self):
        pass

    def results(self, split, model, mcd_df_model, params, num_transformer, cat_transformer, rs, balance = 'n'):
        X = mcd_df_model.drop('classification', axis=1)
        y = mcd_df_model['classification']

        # remove the categorical aspect of y since we don't want it to be treated as a category
        y = y.astype('object')

        # convert X.columns to a list
        # X.col = list(X.columns)

        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()
        print(f'numerical features used are {numerical_features}')
        print(f'categorical features used are {categorical_features}')
        print()

        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=rs, stratify=y)
        
        if num_transformer == None:
            pass
        else:
        # Preprocessing
            preprocessor = ColumnTransformer(transformers=[
                ('tnf1',num_transformer,numerical_features),
                ('tnf3',cat_transformer,categorical_features)
                ])
            
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
        # Do class balancing
        if balance == 'y':
            smote = SMOTE(random_state=rs)
            pipeline = imbPipeline(steps=[('preprocessor', preprocessor),("smote",smote),('classifier', model)])
        elif num_transformer == None:
            pipeline = Pipeline(steps=[('classifier', model)])
        else:
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),('classifier', model)])
        
        grid = GridSearchCV(pipeline, params, cv=skf, scoring ='accuracy', verbose=1)

        # fit model to data
        grid.fit(X_train, y_train)

        # predict the test data
        y_pred = grid.predict(X_test)

        # print the best parameters
        print('The best training parameters and cross validated accuracy score are:')
        print(grid.best_params_)
        print(f'the range of scores can be {cross_val_score(grid.best_estimator_, X_train, y_train, cv=skf)}')
        print(f'{grid.best_score_}')

        y_pred = grid.best_estimator_.predict(X_test)
        print(classification_report(y_test, y_pred))
        print('Balanced Accuracy: {:.2f}\n'.format(balanced_accuracy_score(y_test, y_pred)))

        # print confusion matrix with the predicted and actual correct with colors
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # put labels on the predicted and actual labels
        plt.xticks([0.5,1.5,2.5], ['High', 'Low', 'Medium'])
        plt.yticks([0.5,1.5,2.5], ['High', 'Low', 'Medium'])

        plt.show()
  
        return grid