import urllib
from urllib.parse import urlparse
import httplib2 as http
import json
import requests
import pandas as pd
from geopy.distance import geodesic

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
    
    def traffic_count(self, distance_list, time_list, day_list, mcd_df, bus_df):
        traffic_type = ['in','out']

        # add empty columns to mcd_df for each i in distance_list, day_list and time_list
        results_df = pd.DataFrame(columns=[f'{traffic}_{day}_{time}_{distance}' for traffic in traffic_type for day in day_list for time in time_list for distance in distance_list])

        for mcd in range(len(mcd_df)):
            
            # create a temporary dataframe to store filtered rows so it has the same columns as bus_df
            temp_df = pd.DataFrame(columns=bus_df.columns)
            
            for stop in range(len(bus_df)):

                # calculate the distance between the bus stop and mcd
                dist = geodesic((bus_df['latitude'][stop], bus_df['longitude'][stop]), \
                        (mcd_df['latitude'][mcd], mcd_df['longitude'][mcd])).km
                
                # if the distance is within the stated amount, include it in the temporary dataframe
                if dist <= distance_list[-1]:
                    # add bus_df.iloc[stop,:] to temp_df using concat because append is deprecating
                    temp_df = pd.concat([temp_df, bus_df.iloc[stop,:].to_frame().transpose()], axis=0)
                elif dist <= distance_list[1]:

            # create a new groupby object with the temp_df and group by day_type and time_per_hour        
            temp_df = temp_df.groupby(['DAY_TYPE','TIME_PER_HOUR'])[['in','out']].sum()
            
            # reset index of temp_df to make all the rows filled with the group by values
            temp_df.reset_index(inplace=True)
            
            # melt the df to get the traffic_type, day_type, time_per_hour and traffic_count in 1 row
            melted_df = pd.melt(temp_df, id_vars=['DAY_TYPE', 'TIME_PER_HOUR'], value_vars=['in','out'], var_name='traffic_type', value_name='traffic_count')

            # create a unique column for each traffic_type + '_' + day_type + '_' + time_per_hour + '_' + distance  and get the traffic_count all in 1 row
            pivoted_df = melted_df.pivot(index=None, columns=['traffic_type','DAY_TYPE', 'TIME_PER_HOUR'], values='traffic_count')
            pivoted_df.columns = ['_'.join(map(str, col)).strip() + '_' + str(distance) for col in pivoted_df.columns.values]
            pivoted_df.index.name = None

            # compress pivoted_df to 1 row based on the sum of the traffic_count in each column
            pivoted_df = pivoted_df.sum(axis=0).to_frame().transpose()
            
            # add the pivoted_df to results_df
            results_df = pd.concat([results_df, pivoted_df], axis=0)

            print(f"Done with Mcdonalds {mcd} for distance {distance}")
            print(results_df.shape)
        return results_df
            

            

class EDA_Data:

    def __init__(self, mcd_df):
        # select only numerical columns
        self.num = mcd_df.select_dtypes(include=['float64','int64'])
        self.numcol = mcd_df.select_dtypes(include=['float64','int64']).columns
        
        # select only categorical columns
        self.cat = mcd_df.select_dtypes(include=['object'])
        self.catcol = mcd_df.select_dtypes(include=['object']).columns
        
        self.classification = mcd_df['Classification']
        self.altclass = mcd_df['AltClass'] 
    
        self.total = mcd_df
class Model:

    def __init__(self):
        pass

    def results(self, model, cat_vars, num_vars):
        cat_vars = ['Region Type','Planning Area','Region']
        num_vars = ['NUM_STOPS', 'num_hdb','TOTAL_TAP_OUT_VOLUME','TOTAL_TAP_IN_VOLUME']

class Test:
    def __init__(self):
        pass

    def test(self, matrix_df,i):
        # matrix_df = matrix_df
        print(matrix_df)
        matrix_df.iloc[0,0] = i
        # output_df = output_df.transpose()
        # output_df = output_df.sum(axis=1)
        # output_df = output_df.to_frame()
        return matrix_df

    