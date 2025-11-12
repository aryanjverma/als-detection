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
model, scaler = joblib.load("svm.joblib")
# I used a MLP Classifier from Scikit Learn, using a leave-out training split where I left out one
# nomral sample and one ALS sample for testing, as this was suggested by Dr.Wang to prevent over 
# fitting. SVM was not good enough for my standards getting 70-80% accuracy, but this shallow neural
# network got 94.6% accuracy.  

X_test = scaler.transform(X)
y_pred = model.predict(X_test) 
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)
# Confusion Matrix:
#                Predicted
#               |   0   |   1   |
#        ------------------------
# Actual  0  | 1828 |   4  |    True Negatives (TN) = 1828, False Positives (FP) = 4
#         1  |  87  | 648  |    False Negatives (FN) = 87, True Positives (TP) = 648
#
# Interpretation:
# - The model correctly predicted 1828 negatives and 648 positives.
# - It incorrectly predicted 4 positives that were actually negative (FP).
# - It missed 87 positives that were predicted as negative (FN).
#
# Sensitivity: 0.9978
# Specificity: 0.8816
# Accuracy: 0.9646

sensitivity = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
specificity = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]) 
print("Sensitivity: " + str(sensitivity))
print("Specificity: " + str(specificity))
print("Accuracy: " + str(model.score(X_test, y)))
# Sensitivity: 0.9978165938864629
# Specificity: 0.8816326530612245
# Accuracy: 0.9645500584339696
