from flask import Flask, render_template, request
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset and prepare the model
file = "https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/Bengaluru_House_Data_clean.csv"
df = pd.read_csv(file)

# Drop irrelevant columns and prepare training data
X = df.drop(['Unnamed: 0', 'price'], axis='columns')
Y = df['price']

# Train the model
lrmodel = LinearRegression()
lrmodel.fit(X, Y)

# Define the predictprice function
def predictprice(location, sqft, bath, bhk):
    try:
        loc_index = np.where(X.columns == location)[0][0]
    except IndexError:
        raise ValueError("Location '{}' not found in the data columns.".format(location))  # Fixed syntax

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lrmodel.predict([x])[0]

# Flask routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/project')
def project():
    locations = X.columns[5:]  # Assuming location columns start from index 5
    return render_template("project.html", locations=locations)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        loc = request.form.get("loc")
        sqft = float(request.form.get("size"))
        bhk = int(request.form.get("bhk"))
        bath = int(request.form.get("bath"))

        # Call predictprice function
        price = predictprice(loc, sqft, bath, bhk).round(3)
        locations = X.columns[5:]
        return render_template("project.html", pprice=price, locations=locations)

    except Exception as e:
        locations = X.columns[5:]
        return render_template("project.html", error=str(e), locations=locations)

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/getlocation')
def get_location():
    return render_template("getLocation.html")

if __name__ == '__main__':
    app.run(debug=True)
