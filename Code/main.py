import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
from oct import optimalDecisionTreeClassifier
import os 
import pandas as pd
import time

def load_data():
    data = np.load('2Croppedyalefaces_preprocessed.npz', allow_pickle=True)

    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    print("Shapes of the datasets:")
    print("X_train: ", X_train.shape)
    print("X_val: ", X_val.shape)
    print("X_test: ", X_test.shape)
    print("y_train: ", y_train.shape)
    print("y_val: ", y_val.shape)
    print("y_test: ", y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test
    #return X_train, X_test, y_train, y_test

def main():
    dataset = 'MF-AugCroppedYALE'
    depths = [2, 3]
    alphas = [0, 0.1, 0.01]
    timelimit = 4000

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    classes = np.unique(y_train)
    print("Unique labels in training data: ", classes)
    res_oct = pd.DataFrame(columns=['instance', 'depth', 'alpha', 'train_acc', 'val_acc', 'test_acc', 'gap', 'train_time', 'labels used'])
    num_unique_labels = len(classes)

    print("Starting the training process...")
    for d in depths:
      for a in alphas:
        print(f"Training OCT with depth={d}, alpha={a}")
        octree = optimalDecisionTreeClassifier(max_depth=d, min_samples_split=2, alpha=a, warmstart=False , timelimit=timelimit, output=True)
        
        tick = time.time()
        octree.fit(X_train, y_train)
        tock = time.time()
        train_time = tock - tick
        
        y_train_pred = octree.predict(X_train)
        y_val_pred = octree.predict(X_val)
        y_test_pred = octree.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=classes)
        y_test_pred_bin = label_binarize(y_test_pred, classes=classes)

        # Compute Mean Average Precision Score
        #map_score = average_precision_score(y_test_bin, y_test_pred_bin, average='macro')

        res_oct = res_oct.append({
            'instance': dataset, 
            'depth': d, 
            'alpha': a,
            'train_acc': train_acc, 
            'val_acc': val_acc, 
            'test_acc': test_acc,
            'gap': octree.optgap,
            'train_time': train_time,
            'labels_used': num_unique_labels
        }, ignore_index=True)

        print(f"Results for OCT with depth={d}, alpha={a}:")
        print('Train accuracy:', train_acc, 'Val accuracy:', val_acc, 'Test accuracy:',  test_acc, 'Gap:', octree.optgap, 'Train time:', train_time)

    #rules = octree._getRules()

    print("Saving results to CSV...")
    if os.path.isfile('oct.csv') and os.path.getsize('oct.csv') > 0:
        res_oct.to_csv('oct.csv', mode='a', header=False, index=False)
    else:
        res_oct.to_csv('oct.csv', mode='a', index=False)
    print("Results saved.")

    # Print the rules in a human-readable format
    '''
    for node, rule in rules.items():
        if rule.feat is not None:
            print(f"Node {node}: If feature {rule.feat} <= {rule.threshold} go to node {2*node}, else go to node {2*node + 1}")
        elif rule.value is not None:
            print(f"Node {node}: Predict class {np.argmax(rule.value)}")
        else:
            print(f"Node {node}: No rule")
    '''
if __name__ == "__main__":
    main()
