import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os
from sklearn.metrics import accuracy_score

# Import necessary libraries
from flask import render_template
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('diabetes.csv')

dataset_new = dataset
dataset_new[["Pregnancies" , "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI" , "DiabetesPedigreeFunction", "Age"]] = dataset_new[["Pregnancies","Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]].replace(0, np.NaN)



# Replacing NaN with mean values
dataset_new["Pregnancies"].fillna(dataset_new["Pregnancies"].mean(), inplace = True)
dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)
dataset_new["DiabetesPedigreeFunction"].fillna(dataset_new["DiabetesPedigreeFunction"].mean(), inplace = True)
dataset_new["Age"].fillna(dataset_new["Age"].mean(), inplace = True)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_new)

dataset_scaled = pd.DataFrame(dataset_scaled)


X = dataset_scaled.iloc[:, [1, 2 , 4 , 5 , 7]].values
Y = dataset_scaled.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )

script_dir = os.path.dirname(__file__)

# Set absolute paths for model and data files
model_file_path = os.path.join(script_dir, 'model.pkl')
data_file_path = os.path.join(script_dir, 'diabetes.csv')

app = Flask(__name__, static_url_path='/static')
model = pickle.load(open(model_file_path, 'rb'))

dataset = pd.read_csv(data_file_path)

dataset_X = dataset.iloc[:,[1, 2 , 4 , 5 , 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = np.array(float_features).reshape(1, -1)
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))


# Add this to your Flask app



# Load the pickle files for each model
logreg_model = pickle.load(open('logreg_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
svc_model = pickle.load(open('svc_model.pkl', 'rb'))
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
dectree_model = pickle.load(open('dectree_model.pkl', 'rb'))
ranfor_model = pickle.load(open('ranfor_model.pkl', 'rb'))

# Make predictions on the test set
Y_pred_logreg = logreg_model.predict(X_test)
Y_pred_knn = knn_model.predict(X_test)
Y_pred_svc = svc_model.predict(X_test)
Y_pred_nb = nb_model.predict(X_test)
Y_pred_dectree = dectree_model.predict(X_test)
Y_pred_ranfor = ranfor_model.predict(X_test)

# Calculate accuracy for each model
accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg) * 100
accuracy_knn = accuracy_score(Y_test, Y_pred_knn) * 100
accuracy_svc = accuracy_score(Y_test, Y_pred_svc) * 100
accuracy_nb = accuracy_score(Y_test, Y_pred_nb) * 100
accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree) * 100
accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor) * 100

# Render the analysis.html template with accuracy values
@app.route('/analysis')
def analysis():
    return render_template('analysis.html',
                           accuracy_logreg=accuracy_logreg,
                           accuracy_knn=accuracy_knn,
                           accuracy_svc=accuracy_svc,
                           accuracy_nb=accuracy_nb,
                           accuracy_dectree=accuracy_dectree,
                           accuracy_ranfor=accuracy_ranfor)


@app.route('/visualization')
def visualization():
    return render_template('visualization.html')


if __name__ == "__main__":
    app.run(debug=True)
