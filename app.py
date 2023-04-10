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
import time
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

st.set_page_config(page_title="McDonald's Price Predicor", page_icon='🍔', layout='wide')
st.title("🍔 I'm Modeling It!")

st.header("This website predicts prices of a McDonald's based on the location of the restaurant in SG.")
def main():
    html_temp = """<div style='overflow: auto;-webkit-overflow-scrolling: touch;'>
                    <div class='tableauPlaceholder' id='viz1680360398682' style='position: relative'><noscript><a href='#'><img alt='Story ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Mc&#47;McDonaldsinSingapore_16802273308130&#47;Story&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='McDonaldsinSingapore_16802273308130&#47;Story' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Mc&#47;McDonaldsinSingapore_16802273308130&#47;Story&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-GB' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1680360398682');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1090px';vizElement.style.height='847px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
                    </div>"""
    components.html(html_temp,height=900)

def main2():
    html = """<!-- Google tag (gtag.js) -->
                <script async src="https://www.googletagmanager.com/gtag/js?id=G-5CVHVXNS01"></script>
                <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){dataLayer.push(arguments);}
                gtag('js', new Date());

                gtag('config', 'G-5CVHVXNS01');
                </script>"""
    components.html(html, height = 100)

if __name__ == "__main__":    
    main()
    main2()


st.header("If you were to setup a McDonald's anywhere in Singapore, how should you price the menu?")
postal_code = st.text_input('Key in a Singapore Location or Postal Code', 'Senja Hawker Centre 677632')

with st.spinner(text="Prediction in progress..."):
    time.sleep(1)
#######
try:
    url = f"https://developers.onemap.sg/commonapi/search?searchVal={postal_code}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    data = requests.get(url).json()    
    latitude = data['results'][0]['LATITUDE']
    longitude = data['results'][0]['LONGITUDE']
    # convert to float
    latitude = float(latitude)
    longitude = float(longitude)
    searchval = data['results'][0]['SEARCHVAL']
    address = data['results'][0]['ADDRESS']
    st.write(f'The closest location based on the input selected is {searchval} located at {address}')
    
    # Initialize the map with the given coordinates
    m = folium.Map(location=[latitude, longitude], zoom_start=15)

    # Add a marker at the given coordinates
    folium.Marker([latitude, longitude]).add_to(m)

    # Display the map in Streamlit
    folium_static(m)

except:
    st.write('Please try to enter a valid Singapore location or postal code')
    st.stop()

with st.spinner(text="Prediction in progress..."):
    time.sleep(1)

try:
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
        
        # this function initiates the dataframe based on the length of mcd_df and then computes the number of X within the distance
        # the columns of the item to be counted must be 'latitude' and 'longitude'
        result = single_num_count(i,latitude, longitude,hdb_df)
        hdb_data.append(result)

    a = coord_list[0][281]
    b = coord_list[0][194]
    c = coord_list[0][4]
    d = coord_list[0][290]
    e = coord_list[0][134]
    f = hdb_data[0]
    g = coord_list[0][100]
    h = coord_list[0][292]
    i = coord_list[0][298]
    j = coord_list[0][278]
    k = sz_pop
    l = coord_list[0][287]
    m = coord_list[0][378]
    n = coord_list[0][282]
    o = coord_list[0][108]
    p = coord_list[0][5]
    q = sz_density
    r = coord_list[0][289]
    s = hdb_data[2]
    t = hdb_data[1]
    u = coord_list[0][138]
    v = coord_list[0][382]
    w = coord_list[0][299]
    x = sz_income
    y = coord_list[0][193]

    # combined all the letters into an array for machine learning predicting 
    test = np.array([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y])
    test = test.reshape(1,-1)

    top_result = RF.predict(test)[0]
    # df = pd.read_excel('data/mcdonalds_prices_new.xlsx')
    
    st.success(f"If you owned a McDonald's here, you should set {top_result.upper()} prices.")
    ######
    st.caption("_predictions are based on a 69% accuracy rate without 'Tourist' or 'School'. Read more about how the model works on Github_")
    st.markdown("[View my GitHub Repository](https://github.com/NicsLvl/McPredictor)")

    # # select only big mac upsized meal from df
    # filtered_df = df[df['variable']==top_result]
    # bm = filtered_df[filtered_df['Classification']=='Upsized Meal']
    # big_mac = bm[bm['Menu Item']=='Big Mac']
    # big_mac = big_mac['value'].values[0]
    # print(big_mac)

    # ff = bm[bm['Menu Item']=='Chicken McCrispy (2pc)']
    # ff = ff['value'].values[0]

    # st.write(f'Big Mac Upsized Meal: ${big_mac} 🍔🍟🥤')
    # st.write(f'2pc Chicken McCrispy Upsized Meal: ${ff} 🍗🍗🍟🥤')

except:
    st.write('Please try to enter a valid Singapore location or postal code')
    st.markdown("[View my GitHub Repository](https://github.com/NicsLvl/McPredictor)")
    st.stop()