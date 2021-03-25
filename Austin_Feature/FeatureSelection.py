'''
            Machine Learning Feature Selection
                SENG 474: Data Mining
'''

# Modules
import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest, SelectPercentile, RFE, VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def letterToNumber(labels):
    if(labels[0] == 'A'):
        return 1
    elif(labels[0] == 'B'):
        return 2
    elif(labels[0] == 'C'):
        return 3
    elif(labels[0] == 'D'):
        return 4
    elif(labels[0] == 'E'):
        return 5
    else:
        return None


def preProcessData(fileName):
    
    # Reading in the data.
    data = pd.read_csv(fileName)

    # Dropping the ID and Nutrient columns fron the input features.
    inputFeatures = data.drop(['ID', 'Nutrient'], axis=1)

    # Creating a data frame containing only the prediction label (Nutrient)
    outputLabel = data['Nutrient']
    outputLabel = outputLabel.apply(letterToNumber)

    return inputFeatures, outputLabel

def selectTopK(inputFeatures, outputLabel, XTest, scoreFunction, numberOfFeatures):

    # Defining the Feature Selection.
    featureSelection = SelectKBest(score_func=scoreFunction, k=numberOfFeatures)

    # Applying the feature selection.
    results = featureSelection.fit(inputFeatures, outputLabel)

    newXTrain = results.transform(inputFeatures)
    newXTest = results.transform(XTest)


    return newXTrain, newXTest
    

def selectTopPercent(inputFeatures, outputLabel, XTest, scoreFunction, percentageOfFeatures):
    
    # Defining the Feature Selection
    featureSelection = SelectPercentile(score_func=scoreFunction, percentile=percentageOfFeatures)

    # Applying the feauture selection.
    results = featureSelection.fit(inputFeatures, outputLabel)

    newXTrain = results.transform(inputFeatures)
    newXTest = results.transform(XTest)

    return newXTrain, newXTest

def main():

    # This Try/Except is used for reading the file from the command line.
    try:

        # Grabbing the fileName.
        fileName = sys.argv[1]
    
    except:
        print("Error: An input file was not passed")
        exit()

    # Adjust these variables for change the models.
    numberOfFeaturesToKeep_ANOVA = 13
    percentageOfFeaturesToKeep_ANOVA = 80

    numberOfFeaturesToKeep_MUTUAL = 13
    percentageOfFeaturesToKeep_MUTUAL = 80

    numberOfFeaturesToKeepRFE_RANDOM = 10


    # Adjust the dataset, removing column 1 which contains unique ID values.
    inputFeatures, outputLabel = preProcessData(fileName)

    # Split the data into training and testing data.
    XTrain, XTest, yTrain, yTest = train_test_split(inputFeatures, outputLabel, random_state=0, train_size=0.8)

    # Base line test.
    clfRand = RandomForestClassifier(n_estimators=8, max_samples=None, bootstrap=True, max_features='sqrt', random_state=0)
    clfNerual = MLPClassifier(hidden_layer_sizes=(7), max_iter=1500, alpha=1e-4, solver='sgd', learning_rate_init=.01, random_state=0)
    baseRandomModel = clfRand.fit(XTrain, yTrain)
    baseNerual = clfNerual.fit(XTrain, yTrain)
    print(f'\nRandom Forest Base Model Accuracy: {accuracy_score(yTest, baseRandomModel.predict(XTest))}, Features: {len(XTrain.columns)}')
    print(f'Nerual Network Base Model Accuracy: {accuracy_score(yTest, clfNerual.predict(XTest))}, Features: {len(XTrain.columns)}')

    # Using the Recursive feature elimination (RFE)
    randomRFESelector = RFE(clfRand, n_features_to_select=numberOfFeaturesToKeepRFE_RANDOM)

    # Fitting the RFE Selectors.
    randomRFESelector = randomRFESelector.fit(XTrain, yTrain)

    # Create new input features from the RFE selectors.
    newRFETrain_RANDOM = randomRFESelector.transform(XTrain)
    newRFETest_RANDOM = randomRFESelector.transform(XTest)

    # Fitting the models with the new RFE features.
    rfeRandomModel = clfRand.fit(newRFETrain_RANDOM, yTrain)

    # RFE Results.
    print(f'\nRandom Feature Selection RFE: {accuracy_score(yTest, rfeRandomModel.predict(newRFETest_RANDOM))}, Features: {np.shape(newRFETrain_RANDOM)[1]}')

    # Using Variance Threshold method.
    thresholdSelector = VarianceThreshold()
    thresholdSelector = thresholdSelector.fit(XTrain, yTrain)
    newVarianceXTrain = thresholdSelector.transform(XTrain)
    newVarianceXTest = thresholdSelector.transform(XTest)

    # Fitting the models to the new Variance training data.
    varianceRandomModel = clfRand.fit(newVarianceXTrain, yTrain)
    varianceNerualModel = clfNerual.fit(newVarianceXTrain, yTrain)

    # Variance Results.
    print(f'\nRandom Feature Selection Variance Threshold: {accuracy_score(yTest, varianceRandomModel.predict(newVarianceXTest))}, Features: {np.shape(newVarianceXTrain)[1]}')
    print(f'Nerual Feature Selection Variance Threshold: {accuracy_score(yTest, varianceNerualModel.predict(newVarianceXTest))}')

    # Getting the new X train and test sets from Select top K method.
    # Using the ANOVA (f_classif) and selecting the top 13 features.
    newSelectXTrain, newSelectXTest = selectTopK(XTrain, yTrain, XTest, f_classif, numberOfFeaturesToKeep_ANOVA)

    # Getting the new X train and test sets from the select top percent method.
    # Using the ANOVA (f_classif) and selecting the 80 percent of the features are kept.
    newPercentXTrain, newPercentXTest = selectTopPercent(XTrain, yTrain, XTest, f_classif, percentageOfFeaturesToKeep_ANOVA)

    # Fitting the models to the Select top K results.
    selectRandomModel = clfRand.fit(newSelectXTrain, yTrain)
    selectNerualModel = clfNerual.fit(newSelectXTrain, yTrain)

    print(f'\nRandom Feature Selection Top K model (ANOVA): {accuracy_score(yTest, selectRandomModel.predict(newSelectXTest))}, Features: {np.shape(newSelectXTrain)[1]}')
    print(f'Nerual Feature Selection Top K model (ANOVA): {accuracy_score(yTest, selectNerualModel.predict(newSelectXTest))}')

    # Fitting the models to the Select top percent.
    percentRandomModel = clfRand.fit(newPercentXTrain, yTrain)
    percentNerualModel = clfNerual.fit(newPercentXTrain, yTrain)

    print(f'\nRandom Feature Selection Percent model (ANOVA): {accuracy_score(yTest, percentRandomModel.predict(newPercentXTest))}, Features: {np.shape(newPercentXTrain)[1]}')
    print(f'Nerual Feature Selection Percent model (ANOVA): {accuracy_score(yTest, percentNerualModel.predict(newPercentXTest))}')

    # Getting the new X train and test sets from Select top K method.
    # Using the Mutual information (mutal_info_classif) and selecting the top 13 features.
    newMutualXTrain, newMutualXTest = selectTopK(XTrain, yTrain, XTest, mutual_info_classif, numberOfFeaturesToKeep_MUTUAL)

    # Getting the new X train and test sets from the select top percent method.
    # Using the Mutual information (mutal_info_classif) and selecting the 80 percent of the features are kept.
    newPercentMutXTrain, newPercentMutXTest = selectTopPercent(XTrain, yTrain, XTest, f_classif, percentageOfFeaturesToKeep_MUTUAL)

    # Fitting the models to the Select top K results.
    selectRandomModel = clfRand.fit(newMutualXTrain, yTrain)
    selectNerualModel = clfNerual.fit(newMutualXTrain, yTrain)

    print(f'\nRandom Feature Selection Top K model (Mutual): {accuracy_score(yTest, selectRandomModel.predict(newMutualXTest))}, Features: {np.shape(newMutualXTrain)[1]}')
    print(f'Nerual Feature Selection Top K model (Mutual): {accuracy_score(yTest, selectNerualModel.predict(newMutualXTest))}')

    # Fitting the models to the Select top percent.
    percentRandomModel = clfRand.fit(newPercentMutXTrain, yTrain)
    percentNerualModel = clfNerual.fit(newPercentMutXTrain, yTrain)

    print(f'\nRandom Feature Selection Percent model (Mutual): {accuracy_score(yTest, percentRandomModel.predict(newPercentMutXTest))}, Features: {np.shape(newPercentMutXTrain)[1]}')
    print(f'Nerual Feature Selection Percent model (Mutual): {accuracy_score(yTest, percentNerualModel.predict(newPercentMutXTest))}')
    

    


if __name__ == '__main__':
    main()