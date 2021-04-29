import flask
import sklearn
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask,render_template, request
from flask import Flask,render_template, request,  redirect, url_for


app = Flask(__name__)

model = pickle.load(open('/mohamadarafa87/pythonProject2/blob/master/scrip.pkl','rb'))
modelLR = pickle.load(open('/mohamadarafa87/pythonProject2/blob/master/scripL.pkl','rb'))
modelTR = pickle.load(open('/mohamadarafa87/pythonProject2/blob/master/tree.pkl','rb'))
modelKN = pickle.load(open('/mohamadarafa87/pythonProject2/blob/master/KNeighbor.pkl','rb'))


imgFolder = os.path.join('static','img')
app.config['UPLOAD_FOLDER'] = imgFolder


@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def home():
    if request.method == "POST":
        data1 = request.form['a']
        data2 = request.form['b']
        data3 = request.form['c']
        data4 = request.form['d']
        data5 = request.form['e']
        data6 = request.form['f']
        data7 = request.form['g']
        data8 = request.form['h']
        data9 = request.form['i']
        data10 = request.form['j']
        operation = request.form['operation']
        if operation == 'Rfor':
          arr= np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]])
          rf_probs = model.predict_proba(arr)[:, 1][0]
          arr1 = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]])
          pred = model.predict(arr1)

          rfor = os.path.join(app.config['UPLOAD_FOLDER'],'rfor.svg')
          if pred == 1:
              return render_template('home.html', prediction_text1='High risk of having Heart Attack ( Stroke ) Predicted using Random Forest',
                                 prediction_text2='The probablity of having Heart Attack Predicted by Random Forest : {}'.format(round(rf_probs,4)),
                                 user_image1 = rfor)
          else:
              return render_template('home.html',
                                     prediction_text1='No risk of having Heart Attack (No Stroke ) Predicted using Random Forest',
                                     prediction_text2='The probablity of having Heart Attack Predicted by Random Forest : {}'.format(round(rf_probs,4)),
                                     user_image1=rfor)

        elif operation == 'Lreg':
          arr3 = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]])
          LRpred = modelLR.predict(arr3)
          arr4 = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]])
          LRpred_prba = modelLR.predict_proba(arr4) [:,1][0]
          lreg = os.path.join(app.config['UPLOAD_FOLDER'], 'lreg.svg')
          if LRpred == 1:
              return render_template('home.html', prediction_text3='High risk of having Heart Attack ( Stroke ) Predicted using Logistic Regrission',
                                 prediction_text4='The probablity of having Heart Attack Predicted by Logistic Regrission : {}'.format(round(LRpred_prba,4)),
                                 user_image2 = lreg)
          else:
              return render_template('home.html',
                                     prediction_text3='No risk of having Heart Attack (No Stroke ) Predicted using Logistic Regrission',
                                     prediction_text4='The probablity of having Heart Attack Predicted by Logistic Regrission : {}'.format(round(LRpred_prba,4)),
                                     user_image2=lreg)



        elif operation == 'tree':
            arr5 = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]])
            TRpred = modelTR.predict(arr5)
            arr6 = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]])
            TRpred_prba = modelTR.predict_proba(arr6) [:,1][0]
            dtree = os.path.join(app.config['UPLOAD_FOLDER'], 'dtree.svg')
            if TRpred == 1:
                return render_template('home.html',prediction_text5='High risk of having Heart Attack ( Stroke ) Predicted using Decision Tree',
                                   prediction_text6='The probablity of having Heart Attack Predicted by Decision tree : {}'.format(round(TRpred_prba,4)),
                                   user_image3=dtree)
            else:
                return render_template('home.html',prediction_text5='No risk of having Heart Attack (No Stroke ) Predicted using Decision Tree',
                                       prediction_text6='The probablity of having Heart Attack Predicted by Decision tree : {}'.format(round(TRpred_prba,4)),
                                       user_image3=dtree)


        elif operation == 'KN':
            arr7 = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]])
            KNpred = modelKN.predict(arr7)
            arr8 = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]])
            KNpred_prba = modelKN.predict_proba(arr8)[:, 1][0]
            kneig = os.path.join(app.config['UPLOAD_FOLDER'], 'kneig.svg')
            if KNpred == 1:
                return render_template('home.html', prediction_text7='No risk of having Heart Attack ( Stroke ) Predicted using KNeighbor',
                                   prediction_text8='The probablity of having Heart Attack Predicted by KNeighbor : {}'.format(round(KNpred_prba,4)),
                                   user_image4=kneig)
            else:
                return render_template('home.html',prediction_text7='No risk of having Heart Attack (No Stroke ) Predicted using KNeighbor',
                                       prediction_text8='The probablity of having Heart Attack Predicted by KNeighbor : {}'.format(round(KNpred_prba,4)),
                                       user_image4=kneig)
        else:
            return render_template('home.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.form['upload-file']
        data = pd.read_excel(file)
        return render_template('home.html', data=data.to_html(table_id = "my_id",classes = "'Age', 'Height','Weight','SyPres','DPres','Cholesterol','Glucose','Smoke','Alcohol','Active'"))

if __name__== "__main__":
    app.run(debug=True)

# ------------------

