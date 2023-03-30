# Import Libraries
import streamlit as st
import json
import pandas as pd
import requests
from shapely.geometry import shape, Point
import numpy as np
import re
import pickle
from geopy.distance import geodesic
import streamlit.components.v1 as components

st.set_page_config(page_title="Predict McDonalds Price", page_icon='🍔', layout='wide', initial_sidebar_state='expanded')
st.title("🍔 Predict McDonald's Prices! 🍔")
postal_code = st.text_input('Key in a Singapore Location or Postal Code', 'Bugis Village')
st.caption('_the search might not work if you run more than 250 searches per min_')

#######
url = f"https://developers.onemap.sg/commonapi/search?searchVal={postal_code}&returnGeom=Y&getAddrDetails=N&pageNum=1"
data = requests.get(url).json()    
latitude = data['results'][0]['LATITUDE']
longitude = data['results'][0]['LONGITUDE']
# convert to float
latitude = float(latitude)
longitude = float(longitude)
searchval = data['results'][0]['SEARCHVAL']
st.write(f'The location selected is {searchval} located at {latitude:.2f}, {longitude:.2f}')

# Load data
distance_list = [2,1,0.5,0.2]
with open("pickles/bus_data.pkl", "rb") as f:
    bus_data = pickle.load(f)
with open("pickles/location_df.pkl", "rb") as f:
    location_df = pickle.load(f)
with open("pickles/subzone_names.pkl", "rb") as f:
    subzone_names = pickle.load(f)
with open("pickles/polygons.pkl", "rb") as f:
    polygons = pickle.load(f)
with open("pickles/polygons_coord.pkl", "rb") as f:
    polygons_coord = pickle.load(f)
with open("pickles/RF.pkl", "rb") as f:
    RF = pickle.load(f)

# check which subzone the point is in
point = Point(longitude, latitude)
for j, polygon in enumerate(polygons_coord):
    if polygon.contains(point):
        sz = subzone_names[j]
        break

def single_traffic_count(lat, long, stop_df, bus_data, distance_list):
        
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

# multiply the bus_data 96 x 5082 matrix by a 5082 x 4 (distance_list) matrix to get a 96 x 4 matrix for each Coordinates
coord_list = single_traffic_count(latitude, longitude,location_df,bus_data,distance_list)

hdb_df = pd.read_csv('data/hdb_df.csv')

# import the subzone data
sz_pop_area_df = pd.read_csv('data/sz_pop_area.csv',index_col=0)
# ignore index when importing the income data
sz_income_df = pd.read_csv('data/sz_income.csv')

sz_density = sz_pop_area_df[sz_pop_area_df['subzone']==sz]['density'].values[0]
sz_pop = sz_pop_area_df[sz_pop_area_df['subzone']==sz]['Hse'].values[0]
sz_income = sz_income_df[sz_income_df['SZ']==sz]['sz_income'].values[0]

def single_num_count(distance, latitude, longitude, hdb_df):
        # input a value of "1" in the matrix if the distance between the bus stop and mcd is less than the distance specified
        count=0
        for hdb in range(len(hdb_df)):
            dist = geodesic((hdb_df['latitude'][hdb], hdb_df['longitude'][hdb]), \
                            (latitude, longitude)).km
            if dist < distance:
                count+=1
        
        return count

hdb_data = []
# Get the num_hdb feature data
for i in distance_list[1:]:
    print(f'This is the run for distance {i}km')
    
    # this function initiates the dataframe based on the length of mcd_df and then computes the number of X within the distance
    # the columns of the item to be counted must be 'latitude' and 'longitude'

    result = single_num_count(i,latitude, longitude,hdb_df)
    hdb_data.append(result)
    print(f'Successfully finished for distance {i}km')

a = coord_list[0][28]
b = coord_list[0][378]
c = coord_list[0][26]
d = coord_list[0][194]
e = coord_list[0][22]
f = coord_list[0][286]
g = coord_list[0][122]
h = sz_income
i = coord_list[0][142]
j = coord_list[0][277]
k = coord_list[0][281]
l = sz_pop
m = hdb_data[0]
n = coord_list[0][100]
o = coord_list[0][130]
p = coord_list[0][285]
q = coord_list[0][138]
r = coord_list[0][108]
s = coord_list[0][374]
t = coord_list[0][118]
u = coord_list[0][381]
v = coord_list[0][289]
w = coord_list[0][299]
x = coord_list[0][382]
y = coord_list[0][298]
z = coord_list[0][292]
aa = coord_list[0][42]
ab = coord_list[0][282]
ac = coord_list[0][24]
ad = coord_list[0][274]
ae = coord_list[0][193]
af = sz_density
ag = coord_list[0][5]
ah = coord_list[0][293]
ai = coord_list[0][4]
aj = coord_list[0][302]
ak = coord_list[0][196]
al = hdb_data[2]
am = coord_list[0][377]
an = coord_list[0][129]
ao = coord_list[0][278]
ap = hdb_data[1]
aq = coord_list[0][290]
ar = coord_list[0][132]

# combined all the letters into an array for machine learning predicting 
test = np.array([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao,ap,aq,ar])
test = test.reshape(1,-1)

top_result = RF.predict(test)[0]
df = pd.read_excel('data/mcdonalds_prices.xlsx')

# Display Results
st.subheader('Result')
st.write(f"If you owned a McDonald's here, you would set {top_result} prices. Here's some info to get started:")
######

import streamlit.components.v1 as components
# def main():
#     html_temp = """<div class='tableauPlaceholder' id='viz1680175741167' style='position: relative'><noscript><a href='#'><img alt='Story 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Mc&#47;McDonaldsinSingapore&#47;Story1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='McDonaldsinSingapore&#47;Story1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Mc&#47;McDonaldsinSingapore&#47;Story1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1680175741167');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1016px';vizElement.style.height='991px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
#     components.html(html_temp, height=1000)
# if __name__ == "__main__":    
#     main()