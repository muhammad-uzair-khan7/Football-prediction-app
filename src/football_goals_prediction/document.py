import pandas as pd

# data= pd.read_csv('../../2021-2022 Football Player Stats.csv', encoding="cp1252")

def load_data(data):
    data= pd.read_csv(data, encoding="cp1252")
    return data

if __name__ == "__main__":
    load_data()