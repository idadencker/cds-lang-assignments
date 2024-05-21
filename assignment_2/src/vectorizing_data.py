import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from joblib import dump, load
import scipy as sp
from codecarbon import EmissionsTracker



def prepare_data(filepath, tracker):
    """Start carbon tracker"""
    tracker.start_task("prepare data")
    '''
    Read in data and create X and y split
    '''
    news_data = pd.read_csv(filepath)

    X = news_data["text"]
    y = news_data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X,           
                                                        y,         
                                                        test_size=0.2,   
                                                        random_state=42) 
    '''
    Define vectorizer that only makes unigrams and bigrams, lowercases all, removes very common and rare words and keeps only the top 500 features
    '''
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     
                                lowercase =  True,       
                                max_df = 0.95,           
                                min_df = 0.05,           
                                max_features = 500)      
    '''
    Save the vectorizer in the models folder
    '''
    dump(vectorizer,"models/tfidf_vectorizer.joblib")
    tracker.stop_task()

    return X_train, X_test, vectorizer



def fit_vectorizer(X_train, X_test, vectorizer, tracker):
    """Start carbon tracker"""
    tracker.start_task("fit vectorizer")

    '''
    Fit the vectorizer to the data. fit_transform() is used on the training data to scale it. The test data is transformed.
    This ensures that the model is not biased towards a specific feature and prevents the model from learning from the test data.
    '''
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)

    feature_names = vectorizer.get_feature_names_out()
    '''
    Save the spicy sparse matrix in the feature_extracted_object folder
    '''
    sp.sparse.save_npz('feature_extracted_object/X_train_feats.npz', X_train_feats)
    sp.sparse.save_npz('feature_extracted_object/X_test_feats.npz', X_test_feats)
    tracker.stop_task()



def main():
    tracker = EmissionsTracker(project_name="Assignment_2_vectorizing data",
                           output_dir= os.path.join("..","assignment_5", "out"),
                           output_file="emissions_assignment_2_vectorizing_data.csv")

    filepath= os.path.join("in", "fake_or_real_news.csv")
    X_train, X_test, vectorizer = prepare_data(filepath, tracker)
    fit_vectorizer(X_train, X_test, vectorizer, tracker)
    tracker.stop() 



if __name__ == "__main__":
    main()