from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import random

X_train_temp = []
y_train = []
X_test_temp = []
y_test = []
# Function to generate a link given a number sample and whether they're normal or not
def generate_link(letter, number):
    if number < 10:
        return "ALSDetection_data/" + letter + "0" + str(i) + ".csv"
    else:
        return "ALSDetection_data/" + letter + str(i) + ".csv"
# Turns a line in a csv to a vector of floats 
def line_to_vector(line):
    old_vector = line.split(',')
    new_vector = []
    for element in old_vector:
        new_vector.append(float(element))
    return new_vector
# Picks random pair to use for testing
randomNumber = random.randint(1,11)

# Converts CSVs to matrices for training and testing  
print("Processing data...")
for i in range(1,12):
    a_link = generate_link("A", i)
    n_link = generate_link("N", i)
    if (i != randomNumber):
        with open(a_link, "r") as a:
            all_lines = a.readlines()
            for line in all_lines:
                vector = line_to_vector(line)
                X_train_temp.append(vector)
                y_train.append(1)
        with open(n_link, "r") as n:
            all_lines = n.readlines()
            for line in all_lines:
                vector = line_to_vector(line)
                X_train_temp.append(vector)
                y_train.append(0)
    else:
        with open(a_link, "r") as a:
            all_lines = a.readlines()
            for line in all_lines:
                vector = line_to_vector(line)
                X_test_temp.append(vector)
                y_test.append(1)
        with open(n_link, "r") as n:
            all_lines = n.readlines()
            for line in all_lines:
                vector = line_to_vector(line)
                X_test_temp.append(vector)
                y_test.append(0)
print("Data processsed")
print(str(len(X_train_temp)) + " by " + str(len(X_train_temp[0])) + " matrix")
print("Training model...")
# Scales training data by standarizing data based on mean and variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_temp)
X_test = scaler.transform(X_test_temp)
# The model is a shallow neural network. I used early stopping and alpha L2 regularization to 
# prevent overfitting. 
clf = MLPClassifier(random_state=42, max_iter=1500, hidden_layer_sizes=(64,32), 
                    learning_rate_init=0.0005, early_stopping=True, 
                    alpha=0.01).fit(X_train, y_train)

print("Model trained.")
print("Training score: " + str(clf.score(X_train, y_train)))
print("Testing accuracy: " + str(clf.score(X_test, y_test)))

# This saves the model. I commented it out because I already have a good model.
'''
filename = "model.joblib"
joblib.dump((clf, scaler), filename)
print("Model saved.")
'''