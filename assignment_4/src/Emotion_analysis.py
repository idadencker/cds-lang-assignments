import os 
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from codecarbon import EmissionsTracker 



def extract_emotions(df, tracker):
    """Start carbon tracker"""
    tracker.start_task("extract emotions")
    '''
    A pipeline is specified for easy processing
    '''
    classifier = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base", 
                        return_all_scores=False)
    
    '''
    Text is made into list. Hereafter the text is classified using the pipeline and values are saved
    '''
    text = [str(i) for i in df['Sentence'].tolist()]
    
    emotion_scores = []
    for sentence in tqdm (text, desc = 'Extracting emotions on sentences'):
        scores = classifier(sentence)
        emotion_scores.append(scores[0])
    
    """Appends the items to the dataframe """
    labels = [item['label'] for item in emotion_scores]
    scores = [item['score'] for item in emotion_scores]
    
    df['emotion'] = labels
    df['score'] = scores
    tracker.stop_task()

    return df



def season_rel_freq_for_emotions(df, tracker):
    """Start carbon tracker"""
    tracker.start_task("Plotting relative frequencies")

    """ Setting a style and generating a color palette for the bars """
    sns.set_style("darkgrid")
    num_seasons = len(df['Season'].unique())
    color_palette = sns.color_palette("deep", num_seasons)

    """ For arranging the plots, the number of unique emotions is calculated """
    num_emotions = len(df['emotion'].unique())
    
    """ Calculates the number of rows and columns for subplot grid and creates the subplot grid """
    num_rows = (num_emotions - 1) // 3 + 1
    num_cols = min(num_emotions, 3)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    
    """ Loops through data and creates a bar plot for each emotion """
    for i, emotion_label in enumerate(df['emotion'].unique()):
        emotion_data = df[df['emotion'] == emotion_label]
        counts = emotion_data['Season'].value_counts().sort_index()
        total_count = counts.sum()
        relative_freq = counts / total_count 
        ax = relative_freq.plot(kind='bar', color=color_palette , ax=axs[i // 3, i % 3])
        ax.set_title(f'Relative frequency of {emotion_label} across all seasons')
        ax.xaxis.set_label_text("")
        ax.set_ylabel('Relative Frequency')
    
    """ Hide the empty subplots for nicer layout """
    for i in range(num_emotions, num_rows * num_cols):
        axs[i // 3, i % 3].axis('off')
    
    plt.tight_layout()
    plt.savefig("out/Relative_frequency_of_emotions_across_all_seasons.png")
    plt.show()
    tracker.stop_task()



def emotions_count_for_seasons(df, tracker):
    """Start carbon tracker"""
    tracker.start_task("Plotting counts")

    """ Setting a style and generating a color palette for the bars """
    sns.set_style("darkgrid")
    num_emotions = len(df['emotion'].unique())
    color_palette = sns.color_palette("deep", num_emotions)

    """ For arranging the plots, the number of unique emotions is calculated """
    num_seasons = len(df['Season'].unique())
    
    """ Calculates the number of rows and columns for subplot grid and creates the subplot grid """
    num_rows = (num_seasons - 1) // 3 + 1
    num_cols = min(num_seasons, 3)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    
    """ Loops through data and creates a bar plot for each season """
    for i, season_label in enumerate(df['Season'].unique()):
        season_data = df[df['Season'] == season_label]
        ax = axs[i // 3, i % 3] 
        emotions = season_data['emotion'].value_counts().sort_index()
        ax.bar(emotions.index, emotions.values, color=color_palette)
        ax.set_title(f'Count of emotions in {season_label}')
        ax.xaxis.set_label_text("")
        ax.set_ylabel('Count')
    
    """ Hide the empty subplots for nicer layout """
    for i in range(num_seasons, num_rows * num_cols):
        axs[i // 3, i % 3].axis('off')
    
    plt.tight_layout()
    plt.savefig("out/Count_all_emotions_for_seasons.png")
    plt.show()
    tracker.stop_task()



def main():
    tracker = EmissionsTracker(project_name="Assignment_4",
                        output_dir= os.path.join("..","assignment_5", "out"),
                        output_file="emissions_assignment_4.csv")
    tracker.start_task("Reading in data")                   
    df = pd.read_csv("in/Game_of_Thrones_Script.csv")
    tracker.stop_task()

    df = extract_emotions(df, tracker)
    season_rel_freq_for_emotions(df, tracker)
    emotions_count_for_seasons(df, tracker)
    tracker.stop() 

    
    
if __name__ == "__main__":
    main()