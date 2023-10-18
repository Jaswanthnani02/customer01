import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page title and company logo
st.set_page_config(
    page_title="Interactive Company App",
    page_icon=":rocket:",
    layout="wide"
)

# Company logo
st.image("LOGO.png", use_container_width=True)

# Add a title and description
st.title("Welcome to Your Interactive Company App!")
st.write("Explore different cases and visualize data interactively.")

# Dropdown for selecting cases
selected_case = st.selectbox("Select a Case:", ["Case One", "Case Two", "Case Three"])

# Display specific content based on the selected case
if selected_case == "Case One":
    st.header("Case One Details")
    st.write("Details about Case One goes here.")
elif selected_case == "Case Two":
    st.header("Case Two Details")
    st.write("Details about Case Two goes here.")
else:
    st.header("Case Three Details")
    st.write("Details about Case Three goes here.")

# Interactive chart based on user input (using Matplotlib)
st.header("Interactive Chart")

# Sliders for user input
a = st.slider('A', 1, 100, 25)
b = st.slider('B', 1, 100, 50)
c = st.slider('C', 1, 100, 75)
d = st.slider('D', 1, 100, 10)

# Data for the chart
categories = ['A', 'B', 'C', 'D']
values = [a, b, c, d]

# Create a bar chart using Matplotlib
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='skyblue')
plt.xlabel('Category')
plt.ylabel('Values')
plt.title('Interactive Bar Chart')
plt.xticks(rotation=45)
st.pyplot(plt)

# Add a fun fact
st.sidebar.header("Fun Fact")
st.sidebar.write("Did you know? Streamlit is an amazing tool for creating interactive web applications with Python!")

# Footer
st.markdown("---")
st.write("© 2023 Your Company. All rights reserved.")
