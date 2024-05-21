# Query expansion with word embeddings


## Introduction
This program will use the gensim language model 'glove-wiki-gigaword-50' to extract the 10 most similar words to a search term like the word 'dog', 'mother', 'love' etc., and apply it to a dataset containing 57.650 songs, to determine how many percentage of a given artist's songs contains at least 1 of the 10 words. The results will be printed to the terminal and saved as a txt file in the 'out' folder. 


## Data 
The dataset is the ```Spotify Million Song Dataset``` dataset which consist of 57.650 Spotify Songs. It provides a comprehensive collection of data that can be analysed to gain insights into various aspects of the songs in the Spotify library. More information on the dataset and instructions for downloads can be found [here](https://www.kaggle.com/datasets/joebeachcapital/57651-spotify-songs). 


## Repository overview 
The repository consists of:
- 1 README.md file
- 2 bash scripts
- 1 requirements file
- in folder for storing input data
- out folder for holding the saved results. A few examples are already in the folder
- src folder containing the script implementing query expansion with word embeddings


## Reproducibility 
To make the program work do the following:

1) Clone the repository 
```python
$ git clone https://github.com/idadencker/cds-lang-assignments.git
```
2) Download the Spotify Million Song Dataset_exported.csv and place in the 'in' folder
3) In a terminal set your directory:
```python
$ cd assignment_3
```
4) To create a virtual environment run:
```python
$ source setup.sh
```
5) To run the script and save results insert whatever artist and search term you like (do not worry about lower or uppercase, the program will understand): 
```python
$ source run.sh --artist "ABBA" --search_term "car"
# or use the abbreviations: 
$ source run.sh -a "ABBA" -s "car"
```
The results are saved in the out folder and printed directly in your terminal:
    13.27% of abba's songs contain words related to car


## Summary and discussion
Depending on the artist and chosen search term, results are printed and saved. However there are some limitations which should be taken into consideration before interpreting the results. The model uses word embeddings to reach the most similar words. Word embeddings are dense vector representations of words in a continuous vector space where words with similar meanings are closer to each other. The model has been trained on a very large corpus of data to capture co-occurrences of words. Often the algorithm does a good job at returning meaningful and sensible 'neighbours', however at times the model can act unexpectedly, and return 'neighbours' one would not have intuitively deemed similar. Furthermore, the model often adds suffixes to words or inflects a word into another form, so that car becomes cars, mother becomes mothers etc. <br>
Such behaviour doesn't necessarily mean we should dismiss the model altogether; rather, it emphasises the importance of being cautious when interpreting its results.