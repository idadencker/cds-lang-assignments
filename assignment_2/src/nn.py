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



def fit_predict(X_train_feats, X_test_feats, y_train, y_test):
    '''
    Fit the classifier to the data and get predictions
    '''
    classifier = MLPClassifier(activation = "logistic", 
                            hidden_layer_sizes = (20,), 
                            max_iter=1000, 
                            random_state = 42)


    model= classifier.fit(X_train_feats, y_train)

    '''
    Make and save classification report
    '''
    y_pred = model.predict(X_test_feats)

    classifier_metrics = metrics.classification_report(y_test, y_pred)

    filepath = "out/log_classification_report.txt"

    with open(filepath, 'w') as file:
        file.write(classifier_metrics)


    dump(classifier, "models/MLP_classifier.joblib")




def main():
    filepath= os.path.join("..","cds-language", "data", "fake_or_real_news.csv")
    X_train_feats, X_test_feats, y_train, y_test = prepare_data(filepath)
    fit_predict(X_train_feats, X_test_feats, y_train, y_test)



if __name__ == "__main__":
    main()