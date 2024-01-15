#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:35:36 2020

@author: shelbystowe


Learning about directories and working with files 

"""


import os 

os.getcwd()

print(os.getcwd())



import pandas as pd

excel_file = 'ID_100_light_history.xlsx'
ML_data = pd.read_excel(excel_file)

ML_data.head()