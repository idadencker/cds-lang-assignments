import pandas as pd 
import gensim.downloader as api
import argparse
import string



def get_artist_and_word():
    '''
    Function for getting and saving argparsers
    '''
    parser= argparse.ArgumentParser(description= "Loading and printing an array")
    parser.add_argument("--artist", 
                        "-a",
                        required= True, 
                        help= "put name of artist") 
    parser.add_argument("--search_term", 
                        "-s",
                        required= True, 
                        help= "put a search term") 
    args = parser.parse_args()
    args.artist= args.artist.lower()
    args.search_term= args.search_term.lower()
    
    return args



def list_similar_words(model, chosen_search_term):
    '''
    Using the loaded gensim model, a list is created containing the 10 most similar words to the chosen search term  
    '''
    similar_words_list = []
    words = model.most_similar(chosen_search_term)
    for word, _ in words:
        similar_words_list.append(word)

    return similar_words_list



def calculate_stats(args, model, song_data): 
    '''
    Function that will return name of the artist chosen, the chosen search term, and the percentage of that artist songs that contain word(s) related to the search term
    '''
    name_of_artist = args.artist
    chosen_search_term = args.search_term

    """ A list of the most similar words are created """
    similar_words_list = list_similar_words(model, chosen_search_term)

    """ Removes all punctuation characters from the song texts """
    song_data['text'] = song_data['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    """ Create a dataframe containing the artist's songs and raise an error if the artist is not in the data """
    artist_df = song_data[song_data['artist'].str.lower() == name_of_artist]
    if artist_df.empty:
        raise ValueError(f"Artist '{name_of_artist}' not found in the dataset.")

    """ Calculate n_songs_where_word_from_wordslist_appears """
    songs_already_counted = set()  

    n_songs_where_word_from_wordslist_appears = 0
    for word in similar_words_list:
        for song_text in artist_df['text'].str.lower():
            if word in song_text and song_text not in songs_already_counted:
                n_songs_where_word_from_wordslist_appears += 1
                songs_already_counted.add(song_text)  


    percentage = round(((n_songs_where_word_from_wordslist_appears / len(artist_df)) * 100),2)

    return name_of_artist, chosen_search_term, percentage



def main():
    song_data = pd.read_csv("in/Spotify Million Song Dataset_exported.csv")
    model = api.load("glove-wiki-gigaword-50")
    args = get_artist_and_word()
    name_of_artist, chosen_search_term, percentage= calculate_stats(args, model, song_data)
    with open('out/' + f'{name_of_artist}__{chosen_search_term}.txt', 'w') as f:
        print(f"{percentage}% of {name_of_artist}'s songs contain words related to {chosen_search_term}", file=f)
    print(f"{percentage}% of {name_of_artist}'s songs contain words related to {chosen_search_term}")
    


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print("Error:", e)