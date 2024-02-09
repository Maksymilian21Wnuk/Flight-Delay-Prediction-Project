import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB






# NORMALIZACJA

# Normalization
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_balanced)
# X_test_scaled = scaler.transform(X_test)

# X_train = scaler.fit_transform(X_train)
# X_test_s = scaler.transform(X_test)




# DOBRY PODZIAL NA TEST I TRAIN

def train_test_sets_preparation(df):

	# Splitting into training and testing sets while maintaining class proportions
	X_train, X_test, y_train, y_test = train_test_split(df.drop('DELAY', axis=1), df['DELAY'], test_size=0.2, stratify=df['DELAY'], random_state=42)

	# Splitting the training set again into classes 0 and 1
	X_train_0 = X_train[y_train == 0]
	X_train_1 = X_train[y_train == 1]
	y_train_0 = y_train[y_train == 0]
	y_train_1 = y_train[y_train == 1]

	n = min(len(X_train_0), len(X_train_1))

	# Combining subsets of class 0 and 1 in a 1:1 ratio
	X_train_balanced = pd.concat([X_train_0.sample(n=n, random_state=42), X_train_1.sample(n=n, random_state=42)])
	y_train_balanced = pd.concat([y_train_0.sample(n=n, random_state=42), y_train_1.sample(n=n, random_state=42)])

	# X_train_balanced.to_csv('X_train.csv')
	# X_test.to_csv('X_test.csv')
	# y_train_balanced.to_csv('y_train.csv')
	# y_test.to_csv('y_test.csv')

	return X_train_balanced, X_test, y_train_balanced, y_test




# NA RAZIE NIEWAZNE

def split_csv(input_csv, output_prefix, max_rows_per_chunk):
	# Wczytaj cały plik CSV
	df = pd.read_csv(input_csv, chunksize=max_rows_per_chunk)
	
	# Iteruj przez chunki i zapisz każdy chunk do osobnego pliku CSV
	for i, chunk in enumerate(df):
		chunk.to_csv(f"{output_prefix}_{i}.csv", index=False)

def split_array_to_csv(input_array, output_prefix, max_rows_per_chunk):
	# Konwertuj numpy.ndarray na DataFrame
	df = pd.DataFrame(input_array)
	# Oblicz liczbę części, na jakie należy podzielić DataFrame
	num_chunks = len(df) // max_rows_per_chunk + 1
	# Podziel DataFrame na kawałki i zapisz każdy kawałek do osobnego pliku CSV
	for i, chunk in enumerate(np.array_split(df, num_chunks)):
		chunk.to_csv(f"{output_prefix}_{i}.csv", index=False)





# TE SA NA NIEPODZIELONYCH DANYCH


def GDA(X_train, X_test, y_train, y_test):
	# Initializing the GDA classifier
	clf = GaussianNB()

	# Training the classifier on the training data
	clf.fit(X_train, y_train)

	# Predicting labels for the test data
	y_pred = clf.predict(X_test)

	# Evaluating the classifier's accuracy
	accuracy = accuracy_score(y_test, y_pred)
	print(f'Classification Accuracy: {accuracy:.2f}')

	# Displaying the full classification report
	print('\nClassification Report:')
	print(classification_report(y_test, y_pred))

def SVM(X_train, X_test, y_train, y_test):
    # Initializing the SVM classifier
    clf = SVC()

    # Training the classifier on the training data
    clf.fit(X_train, y_train)

    # Predicting labels for the test data
    y_pred = clf.predict(X_test)

    # Evaluating the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classification Accuracy: {accuracy:.2f}')

    # Displaying the full classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))


def LogisticREG(X_train, X_test, y_train, y_test):
	# Initializing the logistic regression model
	model = LogisticRegression()

	# Different regression parameters
	param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag']}

	# Grid search for finding the best parameters
	grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
	grid_search.fit(X_train, y_train)

	# Extracting the best parameters and best estimator from grid search
	best_params = grid_search.best_params_
	best_model = grid_search.best_estimator_

	# Predicting labels for the test set using the best model
	y_pred = best_model.predict(X_test)

	# Calculating the accuracy of the model
	accuracy = accuracy_score(y_test, y_pred)

	# Printing the best parameters and accuracy
	print("LR best parameters:", best_params)
	print("Accuracy:", accuracy)

def DecisionTree(X_train, X_test, y_train, y_test, max_depth=4):
	# Initializing the decision tree classifier
	clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, criterion='gini')

	# Training the classifier on the training data
	t = clf.fit(X_train, y_train)

	# Predicting labels for the test data
	y_pred = clf.predict(X_test)

	# Evaluating the classifier's accuracy
	accuracy = accuracy_score(y_test, y_pred)
	print(f'Classification Accuracy: {accuracy:.2f}')

	# Displaying the full classification report
	print('\nClassification Report:')
	print(classification_report(y_test, y_pred))

def KNN(X_train, X_test, y_train, y_test):
	# Define the range of values for the parameter k to search
	param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

	# Initialize the KNN classifier
	knn = KNeighborsClassifier()

	# Create the GridSearchCV object
	grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')

	# Fit the model to the training data
	grid_search.fit(X_train, y_train)

	# Display the best parameters
	print("Best Parameters:", grid_search.best_params_)

	# Predict labels for the test data using the best model
	y_pred = grid_search.predict(X_test)

	# Evaluate the classification accuracy
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)

	# Display the full classification report
	print('\nClassification Report:')
	print(classification_report(y_test, y_pred))
















# MODELE NA PODZIELONYCH DANCYH


# zalozenia sa takie, ze z calego znormalizowanego pliku X_train powstaly x_train_1, ..., X_train_7 i analogicznie z X_test. W kazdym jest max milion wierszy. y_train i y_test są całe bez podziałów


from sklearn.linear_model import LogisticRegression

# Function to load data from CSV files
def load_data(set_type, index):
    X = pd.read_csv(f'X_{set_type}_{index}.csv')
    y = pd.read_csv(f'Y_{set_type}.csv')
    start = (index - 1) * 1_000_000
    end = start + len(X)
    y = y.iloc[start:end]
    yield X, y.values






# Initialize SGDClassifier model
model = SGDClassifier()

# Iteratively train the model on divided datasets
for X_test, y_test in load_data('train', range(1, 8))
    model.partial_fit(X_train, y_train)

# Evaluate the model on entire test data
test_accuracy = 0
test_samples = 0
for X_test, y_test in load_data('test', range(1, 8)):
    test_accuracy += model.score(X_test, y_test) * len(X_test)
    test_samples += len(X_test)

print("Overall accuracy on test data:")
print(test_accuracy / test_samples)





# Initialize MLPClassifier model
model = MLPClassifier()

# Iteratively train the model on divided datasets
for X_train, y_train in load_data('train', range(1, 8)):
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))

# Evaluate the model on entire test data
test_accuracy = 0
test_samples = 0
for X_test, y_test in load_data('test', range(1, 8)):
    accuracy = model.score(X_test, y_test)
    test_accuracy += accuracy * len(X_test)
    test_samples += len(X_test)

print("Overall accuracy on test data:")
print(test_accuracy / test_samples)


