import pandas as pd

def get_movies_by_genre(dataframe, genres):
    """
    Filter and return movie titles matching any of the given genres.
    """
    genres = [g.strip().lower() for g in genres]
    matching = dataframe[dataframe['predicted_genre'].str.lower().isin(genres)]
    return matching['title'].tolist()

# Load predicted results
df = pd.read_csv("predicted_genres.csv")

print("\n🎯 Genre-based Movie Search (from predicted genres)")
user_search = input("Do you want to search movies by genre? (yes/no): ").strip().lower()

if user_search == "yes":
    user_input = input("Enter genre(s) (comma-separated): ")  # e.g. "comedy, horror"
    genre_list = [g.strip() for g in user_input.split(",")]

    movie_list = get_movies_by_genre(df, genre_list)

    if movie_list:
        print(f"\n🎬 Found {len(movie_list)} movie(s) matching your genre(s):")
        for title in movie_list[:25]:  # limit output to 25 movies
            print("•", title)
    else:
        print("\n⚠️ No movies found for the given genre(s).")
