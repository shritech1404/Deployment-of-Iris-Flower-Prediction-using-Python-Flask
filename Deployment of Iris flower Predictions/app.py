from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('iris.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("forest_fire.html")


@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict(final)


    if prediction == 0:
        return render_template('forest_fire.html', pred='The value is 0 and the flower is a Setosa')
    elif prediction == 1:
        return render_template('forest_fire.html', pred='The value is 1 and the flower is a Versicolor')
    else:
        return render_template('forest_fire.html', pred='The value is 2 and the flower is a Virginica')


if __name__ == '__main__':
    app.run(debug=True)
