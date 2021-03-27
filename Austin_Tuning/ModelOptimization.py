# Modules.
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score

def randomForest(data):

    # Lists for the paramters to be tuned.
    # Adjust these values for different results and/or add other paramters for testing.
    parameters_To_Test = {
        'n_estimators': [100, 200, 300],
        'criterion':('gini', 'entropy'),
        'max_depth': [20, 30, 50],
        'min_samples_split': [2, 4, 8],
        'max_features': ('auto', 'sqrt', 'log2')
        }
    
    # inputFeatures = data.drop(['ID', 'Nutrient'], axis=1)
    inputFeatures = data.drop(['YOrd'], axis=1)

    # Creating a data frame containing only the prediction label (Nutrient)
    outputLabel = data['YOrd']

    # Splitting the data into 80% Training and 20% Testing.
    X_train, X_test, y_train, y_test = train_test_split(inputFeatures, outputLabel, train_size=0.8)

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


def main():

    # Retreiving file name from command line.
    dataset = sys.argv[1]
    
    # Extracting the file name without the extension.
    length = len(dataset) - 4
    fileName = dataset[:length]

    # Reading the Dataset.
    rawData = pd.read_csv(dataset)

    # Random forest.
    randomForest(rawData)
    



if __name__ == '__main__':
    main()