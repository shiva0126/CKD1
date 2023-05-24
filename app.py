import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import pickle
#from ckd_prediction_ import sc

app = Flask(__name__)# app creation
model = pickle.load(open('lgmodel.pkl', 'rb'))# lloading pkl
scalr = pickle.load(open('scaler.pkl', 'rb'))
def CheckGfr(egfr):
    if egfr >= 90:
        stage ="Stage 1"
        cure=1
        
    elif egfr >= 60:
        stage ="Stage 2"
        cure=2
    elif egfr >= 30:
        stage ="Stage 3"
        cure=3
    elif egfr >= 15:
        stage ="Stage 4"
        cure=4
    elif egfr >= 0:
        stage ="Stage 5"
        cure=5
    else:
        stage = "GFR is negative and not possible"
    return stage,cure
    
@app.route('/')
def home():                                #home page 
    return render_template('front_page.html')
 
@app.route('/home')
def homepage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [[float(x) for x in request.form.values()]]
    # get last value of the list
    egfr = int_features[0][-1]
    print(egfr)
    # remove last value from the list
    int_features[0].pop()
    # egfr = int_features[0][0]
    final_fea=scalr.transform(int_features)
   
   
    #features=sc.transform(int_features)
    prediction = model.predict(final_fea)
    pred2 = model.predict_proba(final_fea)
    output2 = '{0:.{1}f}'.format(pred2[0][1],2)
    stage,cure = CheckGfr(egfr)
    if(prediction[0]==0.):
        output='DOESNT HAVE CKD'
        hasStage= ""
        cure=0
    else:
        output=' HAS CKD'
        hasStage = " and is in {}".format(stage)

    return render_template('result.html', prediction_text='THE PATIENT  {} with the probability {} {}'.format(output, output2,hasStage),cure=cure)

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/algorithm')
def algorithm():
    return render_template('algorithm.html')




if __name__ == "__main__":
    app.run(debug=False)