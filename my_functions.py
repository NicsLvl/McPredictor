import urllib
from urllib.parse import urlparse
import httplib2 as http
import json
import requests
import pandas as pd

class BusStopFile:

    def __init__(self,file_path, stop_df=None):
        self.stop_df = stop_df
        self.file_path = file_path
        self.uri= 'http://datamall2.mytransport.sg/ltaodataservice/'

        with open('api_key.txt', 'r') as f:
            self.api_key = f.read().strip()
        self.method = 'GET'
        self.body = ''
        self.headers = { 'AccountKey' : self.api_key, 'accept' : 'application/json'}
        self.h = http.Http()

    def get_stop_data(self, start, end):
        self.start = start
        self.end = end

        # read the api key from a file
        
        
        # specify the target web page
       
        
        for i in range(self.start,self.end,500): # there are currently only 5000+ bus stops and the lta api limits us by 500 each pull
            
            skip = '?$skip=' + str(i)
            target = urlparse(self.uri + self.file_path + skip)
            print(f'The target url is {target.geturl()}')
            
            response, content = self.h.request(target.geturl(),self.method,self.body,self.headers)
            data = json.loads(content)

            df = pd.DataFrame(data['value'])
            self.stop_df = pd.concat([self.stop_df, df], ignore_index=True)

        return self.stop_df
    
    def get_stop_traffic():
        target = urlparse(self.uri + self.file_path)
        print(f'The target url is {target.geturl()}')
        
        response, content = self.h.request(target.geturl(),self.method,self.body,self.headers)
        data = json.loads(content)

        



        