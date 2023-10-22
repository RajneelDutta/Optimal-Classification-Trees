from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from oct import optimalDecisionTreeClassifier
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer

sns.set_style("whitegrid")

def load_datasets():
    iris = load_iris()
    wine = load_wine()
    breast_cancer = load_breast_cancer()
    return [('Iris', iris.data, iris.target),
            ('Wine', wine.data, wine.target),
            ('Breast Cancer', breast_cancer.data, breast_cancer.target)]

def bagging_OCT(X, y, n_estimators=10):
    models = []

    for i in range(n_estimators):
        print(f"\nTraining model {i+1}/{n_estimators}...")
        X_resample, y_resample = resample(X, y)
        model = optimalDecisionTreeClassifier(max_depth=3, timelimit=3000)
        model.fit(X_resample, y_resample)
        models.append(model)
        print(f"Model {i+1} trained.")
    return models


# Load datasets
datasets = load_datasets()

# Prepare to collect results
results = []
classification_reports = []  
n_datasets = len(datasets)
print(n_datasets)
fig, axs = plt.subplots(nrows=n_datasets, ncols=2, figsize=(20, n_datasets*10))

if n_datasets == 1:
    axs = np.array([axs])

for i, (dataset_name, X, y) in enumerate(datasets):
    assert i < n_datasets

    print(f"\nStarting bagging with OCT on {dataset_name} dataset...")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start timing
    start_time = time.time()
    
    # Train OCT with Bagging
    models = bagging_OCT(X_train, y_train, n_estimators=5)

    # Combine predictions from all models
    predictions = []
    for i, model in enumerate(models):
        print(f"\nMaking predictions with model {i+1}...")
        predictions.append(model.predict(X_test))
        accuracy = accuracy_score(y_test, predictions[-1])
        print(f"Model {i+1} accuracy: {accuracy}")

    predictions = np.array(predictions)

    # Majority voting
    print("\nMaking final predictions with majority voting...")
    y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    accuracy = accuracy_score(y_test, y_pred)

    # Compute precision, recall, F1-score and support
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    classification_reports.append(report_df)
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=axs[i, 0])
    axs[i, 0].set_title(f'Confusion Matrix for {dataset_name}')
    axs[i, 0].set_xlabel('Predicted')
    axs[i, 0].set_ylabel('True')

    scorer = make_scorer(accuracy_score)

    # Then use this scorer in the learning_curve function
    train_sizes, train_scores, valid_scores = learning_curve(models[0], X, y, cv=5, scoring=scorer, train_sizes=np.linspace(0.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)

    axs[i, 1].plot(train_sizes, train_scores_mean, label='Training score')
    axs[i, 1].plot(train_sizes, valid_scores_mean, label='Cross-validation score')
    axs[i, 1].set_title(f'Learning Curves for {dataset_name}')
    axs[i, 1].set_xlabel('Training Set Size')
    axs[i, 1].set_ylabel('Score')
    axs[i, 1].legend()
    
    # End timing
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Get tree depth
    tree_depth = models[0].max_depth
    
    print(f"Bagging OCT final accuracy on {dataset_name}: {accuracy}, Time taken: {time_taken} seconds, Tree depth: {tree_depth}")

    results.append((dataset_name, accuracy, time_taken, tree_depth))

# Write results to CSV
results_df = pd.DataFrame(results, columns=['Dataset', 'Accuracy', 'Time Taken', 'Tree Depth'])
results_df.to_csv('bagging_OCT_results.csv', index=False)
print("\nResults written to 'bagging_OCT_results.csv'")

# Write classification reports to CSV
for i, report_df in enumerate(classification_reports):
    report_df.to_csv(f'bagging_OCT_classification_report_{results[i][0]}.csv', index=False)
    print(f"\nClassification report for {results[i][0]} written to 'bagging_OCT_classification_report_{results[i][0]}.csv'")
