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
from codecarbon import EmissionsTracker 



def prepare_data(filepath, tracker):
    """Start carbon tracker"""
    tracker.start_task("Prepare data log_reg")
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
    tracker.stop_task()

    return X_train_feats, X_test_feats, y_train, y_test



def fitting_model(X_train_feats, X_test_feats, y_train, tracker):
    """Start carbon tracker"""
    tracker.start_task("Fitting log_reg model")

    """ Defining what parameters will be optimsed """
    grid = {
        "C": [1.0, 0.1, 0.01], 
        "penalty": ["l1", "l2", 'elasticnet', None],
        "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        "max_iter": [500]
    }
    logreg = LogisticRegression()
    logreg_cv = GridSearchCV(logreg, grid, scoring='accuracy', cv=10, n_jobs=-1, verbose=2)
    
    print("Tuning hyperparameters...")
    model = logreg_cv.fit(X_train_feats, y_train)
    
    """ Print parameters for each combination tried """
    print("\nParameter combinations tried:")
    means = logreg_cv.cv_results_['mean_test_score']
    stds = logreg_cv.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, logreg_cv.cv_results_['params']):
        print(f"Accuracy: {mean:.4f} (±{std:.4f}) for {params}")

    """ Print the hyperparamerts and accuracy for the best model """
    print("Tuned hyperparameters (best parameters):", logreg_cv.best_params_)
    print("Accuracy:", logreg_cv.best_score_)

    y_pred = model.predict(X_test_feats)
    tracker.stop_task()

    return model, y_pred



def save_metrics(y_test, y_pred, model, tracker):
    """Start carbon tracker"""
    tracker.start_task("Save log_reg metrics")

    '''
    Make and save classification report
    '''
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    filepath = "out/log_classification_report.txt"

    with open(filepath, 'w') as file:
        file.write(classifier_metrics)

    dump(model, "models/log_classifier.joblib")
    tracker.stop_task()



def main():
    tracker = EmissionsTracker(project_name="Assignment_2_log_reg",
                           output_dir= os.path.join("..","assignment_5", "out"),
                           output_file="emissions_assignment_2_log_reg.csv")
    filepath= os.path.join("in", "fake_or_real_news.csv")
    X_train_feats, X_test_feats, y_train, y_test = prepare_data(filepath, tracker)
    model, y_pred = fitting_model(X_train_feats,X_test_feats, y_train, tracker)
    save_metrics(y_test, y_pred, model, tracker)
    tracker.stop() 



if __name__ == "__main__":
    main() 