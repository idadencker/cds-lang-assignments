# Extracting linguistic features using SpaCy


## Introduction
This program contains a script that will extract linguistic information including the relative frequency of nouns, verbs, adjectives, and adverbs per 10,000 words as well as the total number of unique person (PER), location (LOC), and organisation (ORG) entities on text files from the Uppsala Student English Corpus (USE). The extracted linguistic information for each subfolder is formatted to CSV and located in the out folder. Two plots are created. One visualising the average relative frequencies by subfolder, and one visualising the avarge number of unique entities by subfolder. <br>
The results are summarised and discussed.


## Data 
The dataset is the ```The Uppsala Student English Corpus (USE)``` dataset which consists of 1.489 essays written by 440 Swedish university students of English at three different levels. The essays cover set topics of different types. All 'A' files are first-term essays, all 'B' files are second-term essays, and all 'C' files are third-term essays. More information on the dataset and instructions for downloads can be found [here](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457). 


## Repository overview 
The repository consists of:
- 1 README.md file
- 2 bash scripts
- 1 requirements file
- in folder for storing input data
- out folder for holding the saved results
- src folder containing the script for extracting linguistic features 


## Reproducibility 
To make the program work do the following:

1) clone the repository 
```python
$ git clone https://github.com/idadencker/cds-lang-assignments.git
```
2) download the USEcorpus.zip and place the unzipped data in the 'in' folder
3) In a terminal set your directory:
```python
$ cd assignment_1
```
4) To create a virtual environment run:
```python
$ source setup.sh
```
5) To run the script and save results run: 
```python
$ source run.sh
```
14 CSV files and 2 plots will be saved the the out folder 


## Summary and discussion
Upon analysing visualisations of relative frequencies and the utilisation of nouns, verbs, adverbs, and adjectives, a consistent pattern emerges across all terms and essays. Nouns overwhelmingly dominate as the most frequently used part of speech (POS) tag in all essays, followed by verbs and adjectives. Conversely, adverbs consistently exhibit the lowest frequency, averaging approximately 400-600 instances per 10,000 words across all essays.<br> While fluctuations within the POS categories show no discernible patterns based on terms (i.e., A, B, or C), examining the average number of named entities reveals more dispersed patterns. Although "person" lists as the most frequently occurring named entity across most essays, organisations follow closely behind. Notably, in the case of category C1, the average number of persons is notably higher compared to other essays within the same category. However, it is essential to highlight that the averages of named entities are not proportional to the length of the essays and are solely based on counts. Upon reviewing the data source documentation, it is evident that C1 comprises longer essays, which logically accounts for the significant count of 'person' named entities within that group. <br>
Several general limitations should be considered, including the fact that terms (A, B, C) encompass essays covering a diverse range of topics, which may influence the frequency of certain POS as different topics can encourage different POS distributions. Additionally, the C1 category (third-term essays) contains relatively few examples, making it challenging to draw conclusive patterns within the three different terms.
