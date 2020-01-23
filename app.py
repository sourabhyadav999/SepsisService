#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:59:51 2020

@author: baby
"""

from flask import Flask,jsonify,request
import pickle
from flask_cors import CORS, cross_origin
import numpy as np

model = pickle.load(open('sepsis-model.pkl', 'rb'))

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

x=['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
       'HospAdmTime', 'ICULOS']

@app.route('/main',methods=['POST'])
@cross_origin()
def predict():
    d = request.get_json()
    
    p = list(map(float,f"{d['HR']},{d['O2Sat']},{d['Temp']},{d['SBP']},{d['MAP']},{d['DBP']},{d['Resp']},{d['EtCO2']},{d['BaseExcess']},{d['HCO3']},{d['FiO2']},{d['pH']},{d['PaCO2']},{d['SaO2']},{d['AST']},{d['BUN']},{d['Alkalinephos']},{d['Calcium']},{d['Chloride']},{d['Creatinine']},{d['Bilirubin_direct']},{d['Glucose']},{d['Lactate']},{d['Magnesium']},{d['Phosphate']},{d['Potassium']},{d['Bilirubin_total']},{d['TroponinI']},{d['Hct']},{d['Hgb']},{d['PTT']},{d['WBC']},{d['Fibrinogen']},{d['Platelets']},{d['Age']},{d['Gender']},{d['Unit1']},{d['Unit2']},{d['HospAdmTime']},{d['ICULOS']}".split(',')))
    
    t=np.reshape(p,(1,-1))
    result = model.predict(t)
    return jsonify({ 'result': str(result) })
                        
@app.route('/getter')
@cross_origin()
def predicter():
    return jsonify({ 'result': 'sachin'})                        
               

if __name__ == '__main__':    
    app.run(debug=True,port=5000)
