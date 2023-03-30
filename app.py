# Import Libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Set Page configuration
# Read more at https://docs.streamlit.io/1.6.0/library/api-reference/utilities/st.set_page_config
st.set_page_config(page_title='Predict Flower Species', page_icon='🌷', layout='wide', initial_sidebar_state='expanded')

# Set title of the app
st.title('🌷 Predict Flower Species')

# Load data
df = pd.read_csv('iris.csv')

# Set input widgets
st.sidebar.subheader('Select flower attributes')
sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.1)
petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.8)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.2)

# Separate to X and y
X = df.drop('Species', axis=1)
y = df.Species

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = RandomForestClassifier(max_depth=2, max_features=4, n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Generate prediction based on user selected attributes
y_pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

# Display EDA
st.subheader('Exploratory Data Analysis')
st.write('The data is grouped by the class and the variable mean is computed for each class.')
groupby_species_mean = df.groupby('Species').mean()
st.write(groupby_species_mean)
st.bar_chart(groupby_species_mean.T)

# Print input features
st.subheader('Variables in Data Set')
input_feature = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
st.write(input_feature)

# Print predicted flower species
st.subheader('Prediction')
st.metric('Predicted Flower Species is :', y_pred[0], '')