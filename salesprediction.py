# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 21:13:49 2021

@author: Bhanupriya
"""

import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
import numpy as np


st.title('Application to predict sales based on youtube Marketing')

st.sidebar.header("user inputs")

def user_inputs():
    youtube_budget=st.sidebar.number_input("Enter the youtube budget")
    youtubesq=youtube_budget*youtube_budget
    inputdata= {"youtube":youtube_budget, "youtube_Sq":youtubesq }
    data=pd.DataFrame(inputdata,index=[0])
    return data
    
    


inputs=user_inputs()


st.write(inputs)


# build the model

train=pd.read_csv("marketing.csv")

train["youtube_Sq"] = train.youtube*train.youtube

model=smf.ols("np.log(sales)~youtube+youtube_Sq",data=train).fit()

prediction=np.exp(model.predict(inputs))



# print out put

st.write("Estimated Sales for the given youtube Budget would lie in the given range")

from scipy import stats

ci=stats.norm.interval(0.95,prediction)

st.write(ci[0],ci[1])








