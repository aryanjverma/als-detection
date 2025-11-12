from sklearn.metrics import confusion_matrix
import joblib
X = []
y = []

# Function to generate a link given a number sample and whether they're normal or not
def generate_link(letter, number):
    if number < 10:
        return "ALSDetection_data/" + letter + "0" + str(number) + ".csv"
    else:
        return "ALSDetection_data/" + letter + str(number) + ".csv"
# Turns a line in a csv to a vector of floats
def line_to_vector(line):
    old_vector = line.split(',')
    new_vector = []
    for element in old_vector:
        new_vector.append(float(element))
    return new_vector

# Converts CSVs to matrices for training and testing  
print("Processing data...")
for i in range(1,12):
    a_link = generate_link("A", i)
    n_link = generate_link("N", i)
    with open(a_link, "r") as a:
        all_lines = a.readlines()
        for line in all_lines:
            vector = line_to_vector(line)
            X.append(vector)
            y.append(1)
    with open(n_link, "r") as n:
        all_lines = n.readlines()
        for line in all_lines:
            vector = line_to_vector(line)
            X.append(vector)
            y.append(0)
print("Data processsed")
model, scaler = joblib.load("model.joblib")
# I used a MLP Classifier from Scikit Learn, using a leave-out training split where I left out one
# nomral sample and one ALS sample for testing, as this was suggested by Dr.Wang to prevent over 
# fitting. 

X_test = scaler.transform(X)
y_pred = model.predict(X_test) 
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)
# Confusion Matrix:
# 
#                Predicted
#               |  0   |   1  |
#        -----------------------
# Actual  0  | 1805 |  27 |    True Negatives (TN) = 1805, False Positives (FP) = 27
#         1  |  111 | 624 |    False Negatives (FN) = 111, True Positives (TP) = 624
#
# Interpretation:
# - The model correctly predicted 1805 negatives and 624 positives.
# - It incorrectly predicted 27 positives that were actually negative (FP).
# - It missed 111 positives that were predicted as negative (FN).
sensitivity = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
specificity = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]) 
print("Sensitivity: " + str(sensitivity))
print("Specificity: " + str(specificity))
print("Accuracy: " + str(model.score(X_test, y)))
# Sensitivity: 0.9852620087336245
# Specificity: 0.8489795918367347
# Accuracy: 0.9462407479548111
