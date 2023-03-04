import urllib
from urllib.parse import urlparse
import httplib2 as http
import json
import requests
import pandas as pd

class BusStopFile:

    def __init__(self, stop_df, file_path):
        self.stop_df = stop_df
        self.file_path = file_path
        self.uri= 'http://datamall2.mytransport.sg/ltaodataservice/'

    def get_stop_data(self, start, end):
        self.start = start
        self.end = end

        # read the api key from a file
        with open('api_key.txt', 'r') as f:
            api_key = f.read().strip()
        
        # specify the target web page
        method = 'GET'
        body = ''
        headers = { 'AccountKey' : api_key, 'accept' : 'application/json'}
        h = http.Http()
        
        for i in range(self.start,self.end,500): # there are currently only 5000+ bus stops and the lta api limits us by 500 each pull
            
            skip = '?$skip=' + str(i)
            target = urlparse(self.uri + self.file_path + skip)
            print(f'The target url is {target.geturl()}')
            response, content = h.request(target.geturl(),method,body,headers)
            data = json.loads(content)

            df = pd.DataFrame(data['value'])
            self.stop_df = pd.concat([self.stop_df, df], ignore_index=True)

        return self.stop_df
        



        