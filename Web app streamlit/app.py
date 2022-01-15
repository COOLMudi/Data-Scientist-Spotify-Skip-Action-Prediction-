import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from PIL import Image
import time

model= pickle.load(open('dtc.pkl','rb'))

def run():
  c1,c2=  st.columns(2)
  with c1:
    img1 =Image.open('unnamed.png')
    #img1=img1.resize((280,200))
    st.image(img1, use_column_width='auto')
  with c2:
    img2 =Image.open('spotify.png')
    #img2=img2.resize((280,200))  
    st.image(img2, use_column_width='auto')
    
  st.title("Spotify Song Skip Prediction App")
  st.subheader("App will predict that a user will skip a song or not")
  st.write("Created By: Mudit Vyas")

  tf_mini = pd.read_csv('tf_mini.csv')
  web_app_scaling_df = pd.read_csv('web_app_scaling_df.csv')

  col1,col2 = st.columns(2)
  with col1:
    trackid= st.text_input("Track Id")
    sesslen= st.slider("Session Length", min_value=0,max_value=20,step=1)
    no_pause= st.radio("No Pause Before Play:", ("Yes","No"))
    seekf= st.number_input("Seek Forward", min_value=0, max_value=100)
    hod= st.number_input("Hours of the Day", min_value=0, max_value=23)
    context_type= st.selectbox("Context Type", 
                                ('editorial_playlist','user_collection',
                                'radio','personalised_playlist','catlog','charts'))
    start= st.selectbox('Start Reason',('trackdone','fwdbtn','backbtn','remote',
                                        'trackerror','clickrow','playbtn','appload','endplay'))

  with col2:
    context= st.selectbox("Context Switch", ("Yes","No"))
    sesspos= st.slider("Session Position", min_value=0,max_value=20,step=1)
    shuffle= st.radio("Shuffle:", ("Yes","No"))
    seekb= st.number_input("Seek Backward", min_value=0, max_value=100)
    ##datetime.date or a tuple with 0-2 dates
    date= st.date_input("Date")
    prem = st.selectbox("Premium",('Yes',"No"))
    end= st.selectbox('End Reason',('trackdone','fwdbtn','backbtn','clickrow',
                                    'remote','playbtn','logout','endplay'))
  
  form_data = {'session_position' : sesspos,  
                'session_length' : sesslen,
                'track_id' : trackid,
                'context_switch' : context,
                'no_pause_before_play' : no_pause,
                'hist_user_behavior_n_seekfwd' : seekf,
                'hist_user_behavior_n_seekback' : seekb,
                'hist_user_behavior_is_shuffle' : shuffle,
                'hour_of_day' : hod,
                'date' : date,
                'premium' : prem,
                'context_type' :context_type,
                'hist_user_behavior_reason_start' : start,
                'hist_user_behavior_reason_end' : end       }

  form_df = pd.DataFrame([list(form_data.values())], columns=list(form_data.keys()))

  # Merging form_df & track_df into single dataframe.
  session_track_data = pd.merge(form_df, tf_mini, on='track_id', how='left')
  
  # Replacing boolean (True, False) by int32 (1, 0)
  session_track_data.replace(['Yes', 'No'], [1, 0], inplace=True)
  # encoding the mode
  session_track_data['mode'].replace({'major': 1, 'minor': 0 }, inplace=True)
  
  # chaning the date to weekday and droping the date column
  session_track_data["date"] = pd.to_datetime(session_track_data["date"])
  session_track_data['week_day'] = session_track_data["date"].dt.dayofweek
  session_track_data.drop("date", inplace=True, axis=1)
  
  session_track_data.replace(['playbtn', 'remote', 'trackerror', 'endplay', 'clickrow'], 'merged', inplace=True)

  # setting one hot encoding for categorical columns (Nominal Columns)
  One_Hot_Encoder = OneHotEncoder()

  context_type = pd.DataFrame(One_Hot_Encoder.fit_transform(session_track_data[['context_type']]).toarray())
  context_type.columns = One_Hot_Encoder.get_feature_names(['context_type'])

  hist_user_behavior_reason_start = pd.DataFrame(One_Hot_Encoder.fit_transform(session_track_data[['hist_user_behavior_reason_start']]).toarray())
  hist_user_behavior_reason_start.columns = One_Hot_Encoder.get_feature_names(['hub_reason_start']) # hub = hist_user_behavior

  hist_user_behavior_reason_end = pd.DataFrame(One_Hot_Encoder.fit_transform(session_track_data[['hist_user_behavior_reason_end']]).toarray())
  hist_user_behavior_reason_end.columns = One_Hot_Encoder.get_feature_names(['hub_reason_end'])  # hub = hist_user_behavior

  # Concatenate dataframe --> session_track_data + context_type + hist_user_behavior_reason_start + hist_user_behavior_reason_end
  session_track_data = pd.concat([session_track_data, context_type, hist_user_behavior_reason_start, hist_user_behavior_reason_end], axis = 1)

  session_track_data.drop(["context_type", "hist_user_behavior_reason_start", "hist_user_behavior_reason_end", "track_id"], axis = 1, inplace = True)
  
  # drop all highly correlated variables.
  session_track_data.drop(['beat_strength', 'danceability', 'dyn_range_mean'], axis=1, inplace=True)

  web_app_scaling_df.drop(['Unnamed: 0'], axis=1, inplace=True)
  web_app_scaling_df = web_app_scaling_df.append(session_track_data)
  web_app_scaling_df.replace([np.nan], [0], inplace=True)
  web_app_scaling_df.reset_index(drop = True, inplace = True)

  # Scaling
  scaler = StandardScaler()
  for col in web_app_scaling_df.columns:
      if (len(web_app_scaling_df[col].unique()) != 2) :
          web_app_scaling_df[col]= scaler.fit_transform(np.array(web_app_scaling_df[col]).reshape(-1, 1))
  


  prediction = model.predict(web_app_scaling_df.tail(1))[0]
  #y = model.predict()
  #lc = [str(i) for i in y]
  #ans = int("".join(lc))
  if st.button("      Predict     "):
    if prediction == 0:
      with st.spinner('Wait for it...'):
        time.sleep(3)
      st.warning('Song is not skipped')
    else:
      with st.spinner('Wait for it...'):
        time.sleep(3)
      st.success('Song is skipped')

  img =Image.open('music.png')
  img=img.resize((500,300))
  st.image(img)


if __name__ == '__main__':
  run()

