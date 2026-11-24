from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

def audit_data(df):
    if df is None:
        raise ValueError("The DataFrame is empty or None. Please provide a valid DataFrame.")
    df['TG']= (df['Goals'] * df['MP']).round()
    return df

def prepare_data(df):
    df= audit_data(df)
    if df is None:
        raise ValueError("The DataFrame is empty or None. Please provide a valid DataFrame.")
    player_data= df.drop(columns=['Rk', 'Nation', 'Player', 'Comp', 'Squad', 'Age', 'Born', 'Pos'])
    X_player= player_data.drop(columns= ['TG'])
    y_player= player_data['TG']
    X_train, X_test, y_train, y_test= train_test_split(X_player, y_player, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def model_goals(df):
    X_train, X_test, y_train, y_test= prepare_data(df)
    pipeline= Pipeline([
        ('scaler', StandardScaler()),
        ('model', HistGradientBoostingRegressor(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    print("Training score: ",pipeline.score(X_train, y_train))
    print("Testing score: ", pipeline.score(X_test, y_test))
    return pipeline, X_test

def predict_goals(df):
    pipeline, X_test= model_goals(df)
    y_pred= pipeline.predict(X_test).round()
    return y_pred

def model_goals_by_player(df, player_name):
    player_data= df[df['Player']==player_name]
    if player_data is None:
        raise ValueError(f"No data found for player: {player_name}")
    pipeline,_= model_goals(df)
    feature_columns= pipeline.feature_names_in_
    prediction= (pipeline.predict(player_data[feature_columns])).round()
    return prediction

