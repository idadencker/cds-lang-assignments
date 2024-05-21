# Emotion analysis with a pre-trained language model


## Introduction
This program contains a script that will load a csv file of all lines from all 8 seasons of the popular television show Game of Thrones (GoT) to perform some computational text analysis. Using a pretrained sentiment analysis model, the emotion-english-distilroberta-base, an emotion for each line is extracted being either anger, disgust, fear, joy, neutral, sadness or surprise as well as a score ranging from 0-1 indicating the intensity of the extracted emotion. <br>
The model is based on the highly complex "DistilRoBERTa" architecture which is a distilled version of the RoBERTa model, which is itself a variant of the well-known BERT (Bidirectional Encoder Representations from Transformers) model, said to be a state-of-the-art model in natural language processing. <br>
Plots demonstrating the distribution of all emotion labels in each season (8 seasons in total) and the relative frequency of each emotion across all seasons (7 emotions in total) are created and saved. The results are summarised and discussed.


## Data 
The dataset is the ```Game of Thrones Script All Seasons``` dataset which consists of all spoken lines from the popular show Game of Thrones. The dataset has a total of 23911 lines. The dataset can be found and downloaded [here](https://www.kaggle.com/datasets/albenft/game-of-thrones-script-all-seasons?select%253DGame_of_Thrones_Script.csv). 


## Repository overview 
The repository consists of:
- 1 README.md file
- 2 bash scripts
- 1 requirements file
- in folder for storing input data
- out folder for holding the saved results
- src folder containing the script implementing emotion analysis 


## Reproducibility 
To make the program work do the following:

1) clone the repository 
```python
$ git clone https://github.com/idadencker/cds-lang-assignments.git
```
2) download the Game_of_Thrones_Script.csv and place in the 'in' folder
3) In a terminal set your directory:
```python
$ cd assignment_4
```
4) To create a virtual environment run:
```python
$ source setup.sh
```
5) To run the script and save results 
```python
$ source run.sh 
```
2 plots are saved in the out folder


## Summary and discussion
From looking at the first plot 'Count_all_emotions_for_seasons' it is clear that neutral is by far the most expressed emotion across all seasons. For all seasons anger is the second most expressed emotion. Generally a very similar pattern for emotions is displayed for all 8 seasons, meaning that the distribution of emotions is close to identical for all seasons. <br>
Looking at the relative frequencies of the different emotions across seasons, joy and disgust seems to decrease over the seasons, while more dispersed patterns are seen for the other emotions. Interestingly, the relative frequency of sadness notably drops for the last two seasons, 7 and 8. Also fear and surprise is low in the last season compared to the remaining seasons. Generally, most emotions are quite stable for seasons with only a few examples that deviate from the little-fluctuation-pattern. <br>
All emotions are extracted using the emotion-english-distilroberta-base model, which generally does a good job at providing emotion scores. However as it appears on the [Hugging Face documentation](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/blob/main/README.md), the model reached an accuracy of 66% after training. Though it is well above chance level at 14%, it is neither 100% making it quite possible that the model misclassified some lines from the dataset. This limitations should naturally be taken into consideration when interpreting the results. 

