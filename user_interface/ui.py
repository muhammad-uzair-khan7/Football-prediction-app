import streamlit as st
import asyncio
from football_goals_prediction.document import load_data
from football_goals_prediction.goals_prediction import audit_data, prepare_data, model_goals, predict_goals, model_goals_by_player

st.markdown("# Welcome to Football Prediction AI")
def data_forward():
    uploaded_file= st.file_uploader(label="Upload your dataset in CSV format", type=["csv"])
    if uploaded_file is not None:
        try:
            return load_data(uploaded_file)
        except Exception as e:
            st.error(f"Error loading data: {e}")
    return None 

data= data_forward()
if data is None:
    st.stop()

df= audit_data(data)
st.markdown("### Dataset Preview")
st.dataframe(df)

st.divider()

option= st.selectbox("Select what you want to predict", ['Goals', 'Assists', 'Position', 'Saves'])
if option == 'Goals':
    st.markdown("<h3 align='center'> You selected to predict Goals</h3>", unsafe_allow_html=True)
    st.markdown("We will predict the number of goals a player will score based on their performance metrics")
    goal_prediction_type= st.radio("Select the type of prediction you want to make", ['Predict Goals for all players', 'Predict Goals for a specific player'], index=None)

    if goal_prediction_type=='Predict Goals for all players':
        total_goals= predict_goals(df)
        st.dataframe(total_goals)
    elif goal_prediction_type == 'Predict Goals for a specific player':
        player_name= st.selectbox("Select a player", df['Player'])
        predict= st.button("Predict.")
        if predict and player_name:
            player_goals= model_goals_by_player(df, player_name)
            st.markdown(f"This player scored **{player_goals}** goals.")
