import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import scipy as sp
from joblib import dump, load
from sklearn.model_selection import GridSearchCV



def prepare_data(filepath):
    news_data = pd.read_csv(filepath)

    X = news_data["text"]
    y = news_data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X,           
                                                        y,          
                                                        test_size=0.2,   
                                                        random_state=42) 
    '''
    Load in vectorized data
    '''

    X_test_feats = sp.sparse.load_npz('feature_extracted_object/X_test_feats.npz')
    X_train_feats = sp.sparse.load_npz('feature_extracted_object/X_train_feats.npz')
    return X_train_feats, X_test_feats, y_train, y_test



def fitting_model(X_train_feats, X_test_feats, y_train):
    """ Defining what parameters will be optimsed """
    grid = {
        "activation": ['logistic', 'tanh', 'relu'],
        "hidden_layer_sizes": [(20,), (50,), (100,)],
        "solver": ['lbfgs', 'sgd', 'adam'],
        'batch_size': [32, 64, 128],
        "max_iter": [500]
    }
    mlp = MLPClassifier()
    mlp_cv = GridSearchCV(mlp, grid, scoring='accuracy', cv=10, n_jobs=-1, verbose=2)
    
    print("Tuning hyperparameters...")
    model = mlp_cv.fit(X_train_feats, y_train)
    
    """ Print parameters for each combination tried """
    print("\nParameter combinations tried:")
    means = mlp_cv.cv_results_['mean_test_score']
    stds = mlp_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, mlp_cv.cv_results_['params']):
        print(f"Accuracy: {mean:.4f} (±{std:.4f}) for {params}")

    """ Print the hyperparamerts and accuracy for the best model """
    print("Tuned hyperparameters (best parameters):", mlp_cv.best_params_)
    print("Accuracy:", mlp_cv.best_score_)

    y_pred = model.predict(X_test_feats)

    return model, y_pred



def save_metrics(y_test, y_pred, model):
    '''
    Make and save classification report
    '''
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    filepath = "out/NN_classification_report.txt"

    with open(filepath, 'w') as file:
        file.write(classifier_metrics)

    dump(model, "models/MLP_classifier.joblib")



def main():
    filepath= os.path.join("in", "fake_or_real_news.csv")
    X_train_feats, X_test_feats, y_train, y_test = prepare_data(filepath)
    model, y_pred = fitting_model(X_train_feats,X_test_feats, y_train)
    save_metrics(y_test, y_pred, model)



if __name__ == "__main__":
    main()