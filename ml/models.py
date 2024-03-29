import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


# zwracane sa tablice numpy.array, zapisujemy je do csv i pozniej bedziemy dzielic na porcje 
# np.savetxt('X_train_scaled.csv', X_train, delimiter=",")
# np.savetxt('X_test_scaled.csv', X_test, delimiter=",")


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
# 330_000
def split_csv(input_csv, output_prefix, max_rows_per_chunk):
	# Wczytaj cały plik CSV
	df = pd.read_csv(input_csv, chunksize=max_rows_per_chunk)
	# Iteruj przez chunki i zapisz każdy chunk do osobnego pliku CSV
	for i, chunk in enumerate(df):
		chunk.to_csv(f"{output_prefix}_{i}.csv", index=False)

# def split_array_to_csv(input_array, output_prefix, max_rows_per_chunk):
# 	# Konwertuj numpy.ndarray na DataFrame
# 	df = pd.DataFrame(input_array)
# 	# Oblicz liczbę części, na jakie należy podzielić DataFrame
# 	num_chunks = len(df) // max_rows_per_chunk + 1
# 	# Podziel DataFrame na kawałki i zapisz każdy kawałek do osobnego pliku CSV
# 	for i, chunk in enumerate(np.array_split(df, num_chunks)):
# 		chunk.to_csv(f"{output_prefix}_{i}.csv", index=False)





# TE SA NA NIEPODZIELONYCH DANYCH

# just to check accuracy of found grid from searching
def BestModelAccuracy(grid_search, X_test, y_test):
    
	best_params = grid_search.best_params_
	best_model = grid_search.best_estimator_
 
	
	y_pred = best_model.predict(X_test)

	acc = accuracy_score(y_test, y_pred)
	print("Accuracy:", acc)
 
	print(f"Best params: {best_params}")
 
	print('\nClassification Report:')
	print(classification_report(y_test, y_pred))

def Bayesian(X_train, X_test, y_train, y_test):
	# Initializing the GDA classifier
	clf = GaussianNB()

	# Training the classifier on the training data
 
	params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
 
	grid_search = 	GridSearchCV(clf, params, cv = 5, scoring='accuracy')
	grid_search.fit(X_train, y_train)

	BestModelAccuracy(grid_search, X_test, y_test)
 
 

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
    clf.fit(X_train, y_train)

    # Predicting labels for the test data
    y_pred = clf.predict(X_test)

    # Evaluating the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classification Accuracy: {accuracy:.2f}')

    # Displaying the full classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Plotting the decision tree
    #plt.figure(figsize=(12, 8))
    #plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['Class 0', 'Class 1'])
    #plt.show()

def KNN(X_train, X_test, y_train, y_test):
	# Define the range of values for the parameter k to search
	#param_grid = {'n_neighbors': [3, 5]}

	# Initialize the KNN classifier
	# use metric w minkowsky with euclidean distance
	knn = KNeighborsClassifier(metric = 'minkowski', p = 2, n_neighbors=3)

	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)

	accuracy = accuracy_score(y_test, y_pred)
 
	print(f"accuracy: {accuracy}")
	print(classification_report(y_test, y_pred))
 
 	# Create the GridSearchCV object
	#grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', verbose = 10)
	# Fit the model to the training data
	#grid_search.fit(X_train, y_train)
	#BestModelAccuracy(grid_search, X_test, y_test)




def SGD(X_train, X_test, y_train, y_test):
	sgdc = SGDClassifier()
	param_grid = {     
		'alpha': [0.0001, 0.001, 0.01],              
		'max_iter': [1000, 2000, 3000, 5000, 10000, 20000],    
	}

	grid_search = GridSearchCV(estimator=sgdc, param_grid=param_grid, cv=5,n_jobs=1, verbose = 10)

	grid_search.fit(X_train, y_train)
	BestModelAccuracy(grid_search, X_test, y_test)


def ADA(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))



'''
def siec(X_train, X_test, y_train, y_test):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Warstwa ukryta z 64 neuronami i funkcją aktywacji ReLU
        Dense(32, activation='relu'), 
        Dense(16, activation='relu'),  
         
        Dense(8, activation='relu'),   # Druga warstwa ukryta z 32 neuronami i funkcją aktywacji ReLU
        Dense(1, activation='sigmoid')                             # Warstwa wyjściowa z jednym neuronem i funkcją aktywacji sigmoid
    ])

    # Kompilacja modelu
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Trenowanie modelu
    history=model.fit(X_train, y_train, epochs=100, batch_size=328, validation_data=(X_test, y_test))
    #i would like to get waights of this model to the file to use it in the future
    model.save_weights('model.h5')
    # Ocena modelu
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    return model, history
'''

# MODELE NA PODZIELONYCH DANCYH


# zalozenia sa takie, ze z calego znormalizowanego pliku X_train powstaly x_train_1, ..., X_train_7 i analogicznie z X_test. W kazdym jest max milion wierszy. y_train i y_test są całe bez podziałó

# okazuje sie, ze trzban dropnac kolumne z datatype z zbiorow Domina
# drop tez dla Unnamed: 0 
# te dropy z wyzej musz byc dla testu i terningu z osoban
# i teraz mozna normalizwoac 
# w test 

X_test = pd.read_csv("/home/meks/Desktop/danexD/X_test.csv")
X_train = pd.read_csv("/home/meks/Desktop/danexD/X_train.csv")
y_test = pd.read_csv("/home/meks/Desktop/danexD/y_test.csv")
y_train = pd.read_csv("/home/meks/Desktop/danexD/y_train.csv")

X_test = X_test.drop(['ARR_DELAY', 'ARR_TIME', 'Unnamed: 0'], axis = 1)
X_train = X_train.drop(['ARR_DELAY', 'ARR_TIME', 'Unnamed: 0'], axis = 1)



scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(y_test.shape)
print(y_train.shape)
print(X_test.shape)
print(X_train.shape)

model = input("give model name: ")



print("begin:")


match model:
	case 'bayes':
		Bayesian(X_train, X_test, y_train, y_test)

	case 'decision':
		#DecisionTree(X_train, X_test, y_train, y_test, max_depth=2)
		#DecisionTree(X_train, X_test, y_train, y_test, max_depth=4)
		#DecisionTree(X_train, X_test, y_train, y_test, max_depth=8)
		DecisionTree(X_train, X_test, y_train, y_test, max_depth=17)

	case 'sgd':
		SGD(X_train, X_test, y_train, y_test)

	case 'knn':
		KNN(X_train, X_test, y_train, y_test)

	case 'gda':
		GDA(X_train, X_test, y_train, y_test)

	case 'forest':
		RandomForest(X_train, X_test, y_train, y_test)

	case 'ada':
		print("ada")
		ADA(X_train, X_test, y_train, y_test)

#SVM(X_train, X_test, y_train.values.ravel(), y_test.values.ravel())

#SVM(X_train, X_test, y_train.values.ravel(), y_test.values.ravel())
