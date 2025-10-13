from sklearn.neural_network import MLPClassifier
import joblib
X_test = []
y_test = []
def generate_link(letter, number):
    if number < 10:
        return "ALSDetection_data/A0" + str(i) + ".csv"
    else:
        return "ALSDetection_data/A" + str(i) + ".csv"
def line_to_vector(line):
    old_vector = line.split(',')
    new_vector = []
    for element in old_vector:
        new_vector.append(float(element))
    return new_vector
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
model = joblib.load("model.joblib")
print(model.score(X_test,y_test))