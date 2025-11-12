import joblib
import numpy as np
file_name = ""
def line_to_vector(line):
    old_vector = line.split(',')
    new_vector = []
    for element in old_vector:
        new_vector.append(float(element))
    return new_vector
X = []
while True:
    try: 
        file_name = input("Enter the file location you want to test (must be in correct format): ")
        with open(file_name, "r") as a:
            all_lines = a.readlines()
            for line in all_lines:
                vector = line_to_vector(line)
                X.append(vector)
        break
    except:
        print("File must exist")
isSVM = False
while True:
    text = input("Enter 0 to try the SVM, 1 to try the neural net: ")
    if (text == "0"):
        isSVM = True
        break
    elif (text == "1"):
        isSVM = False
        break
    else:
        print("Please enter only 0 or 1")
model, scaler = joblib.load("svm.joblib") if isSVM else joblib.load("model.joblib")
X = scaler.transform(X)
y_pred = model.predict(X)
y_pred = np.array(y_pred)
if (np.mean(y_pred) > 0.5):
    print("The model predicts they have ALS")
else:
    print("The model predicts they don't have ALS")