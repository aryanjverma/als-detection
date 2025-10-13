from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

X_train = []
y_train = []
X_test = []
y_test = []
def generate_link(letter, number):
    if number < 10:
        return "ALSDetection_data/" + letter + "0" + str(i) + ".csv"
    else:
        return "ALSDetection_data/" + letter + str(i) + ".csv"
def line_to_vector(line):
    old_vector = line.split(',')
    new_vector = []
    for element in old_vector:
        new_vector.append(float(element))
    return new_vector
print("Processing data...")
for i in range(1,11):
    a_link = generate_link("A", i)
    n_link = generate_link("N", i)
    with open(a_link, "r") as a:
        all_lines = a.readlines()
        for line in all_lines:
            vector = line_to_vector(line)
            X_train.append(vector)
            y_train.append(1)
    with open(n_link, "r") as n:
        all_lines = n.readlines()
        for line in all_lines:
            vector = line_to_vector(line)
            X_train.append(vector)
            y_train.append(0)
for i in range(11,12):
    a_link = generate_link("A", i)
    n_link = generate_link("N", i)
    with open(a_link, "r") as a:
        all_lines = a.readlines()
        for line in all_lines:
            vector = line_to_vector(line)
            X_test.append(vector)
            y_test.append(1)
    with open(n_link, "r") as n:
        all_lines = n.readlines()
        for line in all_lines:
            vector = line_to_vector(line)
            X_test.append(vector)
            y_test.append(0)
print("Data processsed")
print(str(len(X_train)) + " by " + str(len(X_train[0])) + " matrix")
print("Training model...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
clf = MLPClassifier(random_state=3, max_iter=300, hidden_layer_sizes=(88,32), 
                    learning_rate_init=0.00003).fit(X_train, y_train)
print("Model trained.")
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
filename = "model.joblib"
joblib.dump(clf, filename)
print("Model saved.")