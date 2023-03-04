import urllib
from urllib.parse import urlparse
import httplib2 as http
import json
import requests
import pandas as pd

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
                

        



        