import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Stock Market Close Price Prediction")

# Load dataset
data = pd.read_csv("stockmarket.csv")

st.write("Dataset Preview")
st.write(data.head())

# Select columns
x = data[['Day']]
y = data['Close_Price']

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x)

# Regression graph
st.subheader("Linear Regression Graph")

fig, ax = plt.subplots()

ax.scatter(x, y)
ax.plot(x, y_pred)

ax.set_xlabel("Day")
ax.set_ylabel("Close Price")
ax.set_title("Day vs Close Price Regression")

st.pyplot(fig)

# User input for prediction
st.subheader("Predict Close Price")

day = st.number_input("Enter Day", min_value=1)

if st.button("Predict"):
    
    input_data = pd.DataFrame([[day]], columns=['Day'])
    
    result = model.predict(input_data)
    
    st.write("Predicted Close Price:", result[0])