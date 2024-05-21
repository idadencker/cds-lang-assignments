# Classifying fake news data using logistic regression classifier and neural network classifier


## Introduction
This program contains scripts that will classify a fake news dataset as being 'real' or 'fake' using two different classifiers: a logistic regression classifier and a Multi-Layer Perceptron (MLP) neural network classifier. The MLP classifier is an architecturally more complex feedforward artificial neural network model, consisting of fully connected neurons, capable of learning complex patterns and relationships within data. <br>
Prior to classifying, the predictors (X labels) are vectorized using a term frequency–inverse document frequency (TF-IDF) vectorizer which evaluates how relevant a word is to a document in a collection of documents by multiplying two metrics: how many times a word appears in a document, and the inverse document frequency of the word across a set of documents. The tfidf vectorizer and the vectorized data is saved in respectively 'models' and 'feature_extracted_object' folders for later implementation. The vectorized data is loaded and fitted to the two different classifiers. Grid search hyperparameter tuning is implemented to determine the model with the highest accuracy in predicting the label. 
Model performance of both models are evaluated and scoring metrics are summarised in the saved classification report. The results from both methods are summarised and discussed. 


## Data 
The dataset is the ```Fake or Real News``` dataset which consists of 6335 English text articles with evenly distributed labels 'real' or 'fake'. More information on the dataset and instructions for downloads can be found [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news/data). 


## Repository overview 
The repository consists of:
- 1 README.md file
- 2 bash scripts
- 1 requirements file
- models folder for holding the saved classification models and tfidf vectorizer
- feature_extracted_object folder for holding the vectorized and transformed data
- in folder for storing input data
- out folder for holding the saved results
- src folder containing the scripts for vectorizing data and classification 


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
2 classification reports are saved in the out folder

Please note that the scripts may take some time to execute. You can track information on hyperparameter tuning for the models in the terminal output.


## Summary and discussion
Upon examining the classification reports, both models demonstrate strong performance in accurately classifying the data, achieving accuracies of 88% and 90% for the logistic regression classifier and MLP classifier respectively. Furthermore its evident that both models does equally well in classifying the real and fake examples. <br>
Hyperparameter tuning was implemented for both classifiers to improve performance. 
In optimising the models, hyperparameters are selected based on the most commonly optimised ones for that type of model. <br>
For the logistic regression classifier 4 different regularisation strengths, 6 solvers and 4 options for penalties were set equalising 720 possible fits. All fits including accuracy are printed and the best-performing model, optimised for accuracy on the train set uses the saga solver, l1 penalty, and a regularisation strength of 1.0, resulting in an accuracy of 0.8913%. <br>
For the MLP classifier, 3 different activation functions, 3 different sizes of hidden layers, 3 solvers and 3 batchsizes are considered, equalising 810 fits. All fits including accuracy are printed and the best performing model optimised for accuracy on the train set uses tanh activation function, a batchsize of 64, a hidden layer size of 100, and the sgd solver, resulting in an accuracy of 0.8948% <br>
To lower computational resources maximum iterations are set to 500 for both models. It is possible that for some fits, the maximum number of iterations is reached and iterations stopped though the optimisation has not converged yet. This could be avoided by raising the maximum number of iterrations. Similarly, the complexity of hyperparameter tuning could be expaned even further including more or all combinations of hyperparameters, however at the cost of time and computational resources. <br>
In summary, it is evident that both models perform comparably well, achieving a similar accuracy score. Additionally, the scoring metrics for both models show consistency across the classification of both 'real' and 'fake' labels.