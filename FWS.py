from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

iris_model = joblib.load("models/iris.pkl", "r")
churn_model = joblib.load("models/Churn_model.pkl", "r")

@app.route("/")
def test():
    return  "Hello to you"

@app.route("/api/iris/", methods=["POST"])
def iris():
     data = request.get_json(force=True)
     sepal_length = float(data['sepal length (cm)'])
     sepal_width = float(data['sepal width (cm)'])
     petal_length = float(data['petal length (cm)'])
     petal_width = float(data['petal width (cm)'])
     response = iris_model.predict([[sepal_length, sepal_width, petal_length, petal_width]]).tolist()
     return jsonify(response)

@app.route("/api/churn/", methods=["POST"])
def churn():
     cdata = request.get_json(force=True)
     creditScore = int(cdata['CreditScore'])
     Gender = int(cdata['Gender'])
     Age = int(cdata['Age'])
     Tenure = int(cdata['Tenure'])
     Balance = float(cdata["Balance"])
     NumOfProducts = int(cdata["NumOfProducts"])
     HasCrCard = int(cdata["HasCrCard"])
     IsActiveMember = int(cdata["IsActiveMember"])
     EstimatedSalary = float(cdata["EstimatedSalary"])
     France = int(cdata["France"])
     Germany = int(cdata["Germany"])
     Spain = int(cdata["Spain"])
     response = churn_model.predict([[creditScore, Gender, Age, Tenure, Balance,
                                      NumOfProducts, HasCrCard, IsActiveMember,
                                      EstimatedSalary, France, Germany, Spain]]).tolist()
     return jsonify(response)

if __name__ == "__main__":
    app.run(debug = True)
