# Classifying fake news data using logistic regression classifier and neural network classifer


## Introduction
This program contains scripts that will classify a fake news dataset as being 'real' or 'fake' using 2 different classifiers: a logistic regression classifier and a Multi-Layer Perceptron (MLP) neural network classifier. The MLP classifier is an architecturally more complex feedforward artificial neural network model, consisting of fully connected neurons, capable of learning complex patterns and relationships within data. 
Prior to classifiying, the predictors (X labels) are vectorised using a term frequency–inverse document frequency (TF-IDF) vectoriser which evaluates how relevant a word is to a document in a collection of documents by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents. The tfidf vectorizer and the vectorized data is saved in respectively 'models' and 'feature_extracted_object' folders for later implementation. The vectorised data is loaded and fitted to the two different classifiers. Grid search hyperparameter tuning is implemented to determine the model with the highest accuracy in preciting the label. 
Model performance of both models are evaluated and scoring metrics are summarized in the saved classification report. The results from both methods are summarized and discussed. 


## Data 
The dataset is the ```Fake or Real News``` dataset which consist of 6335 english text articles with evenly distributed labels 'real' or 'fake'. More information on the dataset and instructions for downloads can be found [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news/data). 


## Repository overview 
The repository consists of:
- 1 README.md file
- 2 bash scripts
- 1 requirements file
- models folder for holding the saved classification models and tfidf vectorizer
- feature_extracted_object folder for holding the vectorised and transformed data
- in folder for storing input data
- out folder for holding the saved results
- src folder containing the scripts for vectorising data and classification 


## Reproducibility 
To make the program work do the following:

1) clone the repository 
```python
$ git clone https://github.com/idadencker/cds-lang-assignments.git
```
2) download the real_or_fake_news.csv and place in the 'in' folder
3) In a terminal set your directory:
```python
$ cd assignment_2
```
4) To create a virtual environment run:
```python
$ source setup.sh
```
5) To run the scripts and save results run: 
```python
$ source run.sh
```
2 classfication reports are saved in the out folder

Please note that the scripts may take some time to execute. You can track information on hyperparameter tuning for the models in the terminal output.


## Summary and discussion
Upon examining the classification reports, both models demonstrate strong performance in accurately classifying the data, achieving accuracies of 88% and 89% for the logistic regression classifier and MLP classifier respectively. Furthermore its evident that both models does equally well in classifying the real and fake examples. 
Hyperparameter tuning was implemented for both classifiers to improve performance. 
In optimizing the models, hyperparameters are selected based on the most commonly optimized ones for that type of model. 
For the logistic regression classifier 4 different regularization strenghts, 6 solvers and 4 options for penalties were set equalising 720 possible fits. All fits including accuracy are printed and the best-performing model, optimized for accuracy on the train set uses the saga solver, l1 penalty, and a regularization strength of 1.0, resulting in an accuracy of 0.8913%. 
For the MLP classifier, 3 different activation functions, 3 different sizes of hidden layers, 3 solvers and 3 batchsizes are considered, yielding 810 fits. All fits including accuracy are printed and the best performing model optimized for accuracy on the train set uses tahn activation function, a batchsize of 64, a hidden layer size of 100, and the sgd solver, resulting in an accuracy of 0.8948%
To lower computational resoures maximum iterations are set to 500 for both models. it's possible that for some fits, the maximum number of iterations is reached and iterations stopped though the optimization has not converged yet. This could be avoided by raising the maximum number of itterrations. Similarly, the complexity of hyperparameter tuning could be expaned even further including more or all combinations of hyperparameters, however at the cost of time and computational resources. 
In summary, it is evident that both models perform comparably well, achieving an accuracy of approximately 0.88. Additionally, the scoring metrics for both models show consistency across the classification of both 'real' and 'fake' labels.