import streamlit as st 
import pandas as pd
import numpy as np

st.title("🌨️Data prediction⛅")
# Voyons comment sont les données aprés encodage
df= pd.read_csv('weather.csv')
# Nous allons traiter les données manquents
df['Sunshine'].fillna(0, inplace=True)
df['WindGustDir'].fillna(0, inplace=True)
df['WindGustSpeed'].fillna(0, inplace=True)
df['WindSpeed9am'].fillna(0, inplace=True)
df['WindDir3pm'].fillna(0, inplace=True)
df['WindDir9am'].fillna(0, inplace=True)

df.drop('RISK_MM', inplace=True,axis=1)
df.drop('WindGustDir', inplace=True,axis=1)
df.drop('WindDir9am', inplace=True,axis=1)
df.drop('WindDir3pm', inplace=True,axis=1)

# Nous utiliserons LabelEncoder pour gérer les étiquettes de nos principales prévisions
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['RainToday'] = LE.fit_transform(df['RainToday'])
df['RainTomorrow'] = LE.fit_transform(df['RainTomorrow'])

# separation des données 
from sklearn.model_selection import train_test_split
X = df.drop('RainTomorrow',axis=1).values
y = df['RainTomorrow'].values

# standardiser nos informations, car elles sont à des échelles très différentes
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
X = minmax.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
# Passons aux algorithmes prédicteurs svm et xgboost

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model = SVC()
model.fit(X_train, y_train)
previsor_svc = model.predict(X_test)

# les sidebar qui sont dans le Home page  
url="nmamadousow@gmail.com"
with st.sidebar:
    st.info('Data analysis and visualisation')
    st.markdown(''' 
    📧 [Openwather](<https://openweathermap.org/>)
    
    📧 [Météo Sénégal](<https://www.meteoart.com/africa/senegal?page=day#date=2023-07-18>)
    
    💻 [Code source](<nmamadousow@gmail.com>)
                ''')
st.sidebar.info("Xgboost(Extrem Gradient Boosting) accuracy error")
st.sidebar.write('👍Accuracy:',np.round(accuracy_score(y_test,previsor_svc),2)*100,'%')
st.sidebar.write('👎Error:',100-np.round(accuracy_score(y_test,previsor_svc),2)*100,'%')
st.sidebar.header("🌐Data legend")
st.sidebar.success('1 🌨️🌨️🌨️')
st.sidebar.success('0 ⛅⛅⛅')
st.sidebar.write('Made with 💗 by  ☎️ Contact',url)
st.sidebar.info('GOMYCODE SN')

#from xgboost import XGBClassifier
#bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
#bst.fit(X_train,y_train)
#pred_xgb = bst.predict(X_test)
#st.sidebar.write('Accuracy:', np.round(accuracy_score(y_test,pred_xgb),3)*100,'%')

# Fonction input user
def user_input(MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustSpeed,WindSpeed9am,WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday):
    data = np.array([
                     MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustSpeed,WindSpeed9am,
                     WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,
                     Pressure3pm,Cloud9am,Cloud3pm,Temp9am,Temp3pm,RainToday
                     ])
    prediction_data = model.predict(data.reshape(1,-1))
    return prediction_data

# L'utilisateur saisie une valeur pour chaque caracteristique
st.info('You must be enter the values number')
MinTemp = st.number_input('MinTemp:',min_value=-1,value=50)
MaxTemp = st.number_input('MaxTemp:',value=50)
Rainfall = st.number_input('Rainfall:',value=40)
Evaporation = st.number_input('Evaporation:',value=15)
Sunshine = st.number_input('Sunshine:',value=15)
WindGustSpeed = st.number_input('WindGustSpeed:',value=1100)
WindSpeed9am = st.number_input('WindSpeed9am:',value=100)
WindSpeed3pm = st.number_input('WindSpeed3pm:',value=100)
Humidity9am = st.number_input('Humidity9am:',value=100)
Humidity3pm = st.number_input('Humidity3pm:',value=100)
Pressure9am = st.number_input('Pressure9am:',value=1500)
Pressure3pm = st.number_input('Pressure3pm:',value=1500)
Cloud9am = st.number_input('Cloud9am:',value=10)
Cloud3pm = st.number_input('Cloud3pm:',value=10)
Temp9am = st.number_input('Temp9am:',value=30)
Temp3pm = st.number_input('Temp3pm:',value=30)
RainToday = st.number_input('RainToday:',min_value=0,max_value=1)

# creation du bouton de prediction 
if st.button('▶️'):
    prediction = user_input(MinTemp,MaxTemp,Rainfall,Evaporation,Sunshine,WindGustSpeed,WindSpeed9am,
                            WindSpeed3pm,Humidity9am,Humidity3pm,Pressure9am,Pressure3pm,Cloud9am,Cloud3pm,
                            Temp9am,Temp3pm,RainToday)
    st.success(prediction)
#Creation de bot
st.title('🤖Chatbot🤖')
def main():
    #st.write("Bonjour!Je suis un chatbot. Demandez-moi quoi que ce soit sur le sujet dans le fichier texte.")
    # Obtenir la question de l'utilisateur
    question = st.text_input("👨")
    reponse=""
    #Créer un bouton pour soumettre la question
    if st.button("🆗"):
        if question == 'bonjour':
             st.write("🤖",question)  
        elif question == 'donne moi la meteo':
            reponse = "renseignez les données"
            st.write("🤖",reponse)   
if __name__ == "__main__":
    main()

