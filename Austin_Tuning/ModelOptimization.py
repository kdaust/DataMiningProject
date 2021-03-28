# Modules.
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def randomForest(inputFeatures, outputLabel, trainingSize):

    # Lists for the paramters to be tuned.
    # Adjust these values for different results and/or add other paramters for testing.
    parameters_To_Test = {
        'n_estimators': [100, 200, 300],
        'criterion':('gini', 'entropy'),
        'max_depth': [20, 30, 50],
        'min_samples_split': [2, 4, 8],
        'max_features': ('auto', 'sqrt', 'log2')
        }
    
    # Splitting the data into 80% Training and 20% Testing.
    X_train, X_test, y_train, y_test = train_test_split(inputFeatures, outputLabel, train_size=trainingSize)

    # Initialize a random forest model.
    random_forest_model = RandomForestClassifier(random_state=0)

    # Initializing the Grid Search Cross Validation.
    grid_Search = GridSearchCV(random_forest_model, parameters_To_Test, scoring='accuracy')

    # Fit the Grid.
    grid_Search.fit(X_train, y_train)

    print("Best Parameters:")
    print(grid_Search.best_params_)
    print(f'Training Accuracy: {grid_Search.best_score_}')

    y_prediction = grid_Search.predict(X_test)
    print(f'Testing Accuracy: {accuracy_score(y_test, y_prediction)}')

def svm(inputFeatures, outputLabel, trainingSize):

    # Lists for the paramters to be tuned.
    # Adjust these values for different results and/or add other paramters for testing.
    parameters_To_Test = {
        'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]
    }
    
    # Splitting the data into 80% Training and 20% Testing.
    X_train, X_test, y_train, y_test = train_test_split(inputFeatures, outputLabel, train_size=trainingSize)

    # Initialize a nerual network model.
    svm_model = SVC(random_state=0, kernel='rbf')
    # Initializing the Grid Search Cross Validation.
    grid_Search = GridSearchCV(svm_model, parameters_To_Test, scoring='accuracy')

    # Fit the Grid.
    grid_Search.fit(X_train, y_train)

    print("Best Parameters:")
    print(grid_Search.best_params_)
    print(f'Training Accuracy: {grid_Search.best_score_}')

    y_prediction = grid_Search.predict(X_test)
    print(f'Testing Accuracy: {accuracy_score(y_test, y_prediction)}')

def main():

    # Setting the percentage that we want in the training set.
    trainingSize = 0.8

    # Retreiving file name from command line.
    dataset = sys.argv[1]
    
    # Extracting the file name without the extension.
    length = len(dataset) - 4
    fileName = dataset[:length]

    # Reading the Dataset.
    rawData = pd.read_csv(dataset)

    inputFeatures = rawData.drop(['ID', 'Nutrient'], axis=1)
    # inputFeatures = rawData.drop(['YOrd'], axis=1)

    # Creating a data frame containing only the prediction label (Nutrient)
    outputLabel = rawData['Nutrient']
    #outputLabel = rawData['YOrd']

    # Random forest.
    # randomForest(inputFeatures, outputLabel, trainingSize)

    # Nerual Network
    nerualNetwork(inputFeatures, outputLabel, trainingSize)
    



if __name__ == '__main__':
    main()