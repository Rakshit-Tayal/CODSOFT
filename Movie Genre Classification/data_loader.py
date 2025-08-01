import pandas as pd

def load_train_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) == 4:
                _, title, genre, desc = parts
                data.append([title.strip(), genre.strip(), desc.strip()])
    return pd.DataFrame(data, columns=['title', 'genre', 'description'])

def load_test_data(filepath, has_genre=False):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(":::")
            if has_genre and len(parts) == 4:
                _, title, genre, desc = parts
                data.append([title.strip(), genre.strip(), desc.strip()])
            elif not has_genre and len(parts) == 3:
                _, title, desc = parts
                data.append([title.strip(), desc.strip()])
    if has_genre:
        return pd.DataFrame(data, columns=['title', 'genre', 'description'])
    else:
        return pd.DataFrame(data, columns=['title', 'description'])
