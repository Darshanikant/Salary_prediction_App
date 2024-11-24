
# import library
import streamlit as st
import numpy as np
import pickle

# load the pickel file to model
model=pickle.load(open(r"salary price.pkl","rb"))

# write a titel
st.title("SALARY PREDICTION App")

#write some text
st.write("This app provide salary prediction based on your experience .. please enter your year of experience to get the salary prediction..")

# take input from user salary

year=st.number_input("Enter your Experience here..",min_value=0.0, max_value=40.00,value=0.0,step=1.0)

#creat a button
if st.button("Predict Salary"):
    #predict
    exp_input=np.array([[year]]) # creat array
    prediction=model.predict(exp_input) # predict
    
    #return display
    st.success(f"The pridicted ssalary of {year} is: ${prediction}.")
    st.write("Thanks for using the model...Hope you got the output..")
