from crypt import methods
from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask(__name__)

model1 = pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello_world():
    return render_template('predictions.html')

@app.route('/predict', methods=['POST','GET'])

def predict():
    print(request.form)
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model1.predict_proba(final)
    output = '{0:{1}f}'.format(prediction[0][1],2)
    
    return render_template('predictions.html', pred='Price is {}'.format(output))


if __name__ == '__main__':
    app.run()     