from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import spotipy.oauth2 as oauth2
import pandas as pd
import spotipy
import config
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client_credentials_manager = SpotifyClientCredentials(
    client_id=config.client_ID, client_secret=config.client_secret
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def extract_playlist_URI(playlist_URL):
    # Extract the playlist URI from the playlist URL
    playlist_URI = playlist_URL.split("/")[-1].split("?")[0]
    return playlist_URI


def get_tracks_info(playlist_URL):
    # Get the playlist URI
    playlist_URI = extract_playlist_URI(playlist_URL)

    # Get the tracks of the playlist
    playlist_tracks = sp.playlist_tracks(playlist_URI)

    tracks_data = []
    for track in playlist_tracks["items"]:
        playlist_data = {}

        # Get the URI of the track
        track_uri = track["track"]["uri"]
        playlist_data.update({"track_uri": track_uri})

        # Get the name of the track
        playlist_data.update({"track_name": track["track"]["name"]})

        # Get the main artist URI
        artist_uri = track["track"]["artists"][0]["uri"]
        playlist_data.update({"artist_uri": artist_uri})

        # Get information about the main artist
        artist_info = sp.artist(artist_uri)

        # Get the name, popularity, and genres of the artist
        playlist_data.update({"artist_name": track["track"]["artists"][0]["name"]})
        playlist_data.update({"artist_popularity": artist_info["popularity"]})
        playlist_data.update({"genres": artist_info["genres"]})

        # Get the album information
        playlist_data.update({"album_name": track["track"]["album"]["name"]})
        playlist_data.update({"album_uri": track["track"]["album"]["uri"]})
        playlist_data.update({"release_date": track["track"]["album"]["release_date"]})

        # Get the popularity of the track
        playlist_data.update({"track_popularity": track["track"]["popularity"]})

        # Get the audio features of the track
        audio_features = sp.audio_features(track_uri)
        playlist_data.update(audio_features[0])

        keys_to_delete = ["type", "id", "uri", "track_href", "analysis_url"]

        # Remove unnecessary keys from the playlist data
        for key in keys_to_delete:
            if key in playlist_data:
                del playlist_data[key]

        # Add the playlist data to the tracks data list
        tracks_data.append(playlist_data)

    return tracks_data


def vectorize_genres(data_df):
    # Create a copy of the input dataframe
    df = data_df.copy()

    # Create an instance of TfidfVectorizer
    tfidf = TfidfVectorizer()

    # Apply TF-IDF vectorization on the 'genres' column
    tfidf_matrix = tfidf.fit_transform(df["genres"].apply(lambda x: " ".join(x)))

    # Convert the TF-IDF matrix to a DataFrame
    genre_df = pd.DataFrame(tfidf_matrix.toarray())

    # Set column names for the genre DataFrame
    genre_df.columns = ["genre" + "|" + i for i in tfidf.get_feature_names_out()]

    # Reset the index of the genre DataFrame
    genre_df.reset_index(drop=True, inplace=True)

    # Concatenate the original dataframe and the genre DataFrame horizontally
    final_df = pd.concat([df, genre_df], axis=1)

    # Drop the original genres column.
    final_df = final_df.drop("genres", axis=1)

    # Return the final dataframe
    return final_df


def extract_decade(date_str):
    year = int(date_str.split("-")[0])
    decade = (year // 10) * 10
    return int(decade)


def preprocess_decade(data_df):
    # Make a copy of the original DataFrame
    df = data_df.copy()

    # Apply the function to the "release_date" column
    df["release_date"] = df["release_date"].apply(extract_decade)

    return df


def create_summarized_data(playlist_data):
    # Calculate the mean of each column
    summarized_data = playlist_data.mean(numeric_only="True").to_frame(name="mean_value")

    return summarized_data


def align_dataframes(playlist_database, tracks_database):
    # Set a copy of the dataframes
    playlist_data = playlist_database.copy()
    tracks_data = tracks_database.copy()

    # Find missing columns in playlist_data
    missing_columns = list(set(tracks_data.columns) - set(playlist_data.columns))

    # Create a DataFrame with missing columns and default value of zero
    missing_columns_df = pd.DataFrame(0, index=playlist_data.index, columns=missing_columns)

    # Concatenate missing_columns_df with playlist_data
    playlist_data = pd.concat([playlist_data, missing_columns_df], axis=1)

    # Find missing columns in tracks_database
    missing_columns = list(set(playlist_data.columns) - set(tracks_data.columns))

    # Create a DataFrame with missing columns and default value of zero
    missing_columns_df = pd.DataFrame(0, index=tracks_data.index, columns=missing_columns)

    # Concatenate missing_columns_df with tracks_database
    tracks_data = pd.concat([tracks_data, missing_columns_df], axis=1)

    return playlist_data, tracks_data


def generate_recommendations(playlist_data, tracks_database, recommendation_num=30):
    # Create a copy of the database, to avoid modifying it
    tracks_data = tracks_database.copy()

    # Get only the numeric features
    playlist_vector = playlist_data.drop('track_uri', axis=1).values
    track_vectors = tracks_data.drop('track_uri', axis=1).values

    # Normalize the features before making the recommendations
    scaler = MinMaxScaler()
    track_vectors = scaler.fit_transform(track_vectors)
    playlist_vector = scaler.transform(playlist_vector)

    # Calculate cosine similarity between the playlist and the complete song set
    similarity_scores = cosine_similarity(track_vectors, playlist_vector)[:, 0]
    tracks_data['similarity'] = similarity_scores

    # Get the track URIs of the songs in the playlist
    playlist_track_uris = playlist_data['track_uri'].tolist()

    # Filter out tracks that are already on the playlist
    tracks_data = tracks_data[~tracks_data['track_uri'].isin(playlist_track_uris)]

    top_recommendations = tracks_data.sort_values('similarity', ascending=False).head(recommendation_num)
    return top_recommendations.index.to_list()



def get_recommendations_info(recommendations_id, tracks_database):
    # Filter 'tracks_database' DataFrame based on indices from 'recommendations_id'
    track_uris = tracks_database.loc[recommendations_id, "track_uri"]

    # Initialize lists to store the track information
    track_names = []
    track_artists = []
    track_albums = []
    track_urls = []

    # Retrieve track information for each URI
    for track_uri in track_uris:
        track_info = sp.track(track_uri)

        # Extract the desired information
        name = track_info["name"]
        artists = ", ".join([artist["name"] for artist in track_info["artists"]])
        album = track_info["album"]["name"]
        url = track_info["external_urls"]["spotify"]

        # Append the information to the respective lists
        track_names.append(name)
        track_artists.append(artists)
        track_albums.append(album)
        track_urls.append(url)

    # Create a DataFrame to store the track information
    track_info_df = pd.DataFrame(
        {
            "Name": track_names,
            "Artist": track_artists,
            "Album": track_albums,
            "URL": track_urls,
        }
    )

    return track_info_df


def load_database():
    return pd.read_csv(r"C:\Users\yakin\OneDrive - Universidad de Guanajuato\Documentos\Universidad de Guanajuato\8vo Semestre\Análisis de Datos\Proyecto Final\Flask_Web\model_data.csv")


def pipeline(URL):
    data = get_tracks_info(URL)
    data = pd.DataFrame(data)
    dropped_df = data.drop(["artist_name", "artist_uri", "track_name", "album_uri", "album_name"], axis=1)
    preprocess_data = vectorize_genres(dropped_df)
    preprocess_data = preprocess_decade(preprocess_data)
    summarized_data = create_summarized_data(preprocess_data).T

    tracks_database = load_database()
    aligned_playlist_data, aligned_tracks_database = align_dataframes(summarized_data, tracks_database)
    recommendations = generate_recommendations(aligned_playlist_data, aligned_tracks_database)
    recommendations_info = get_recommendations_info(recommendations, tracks_database)

    return recommendations_info