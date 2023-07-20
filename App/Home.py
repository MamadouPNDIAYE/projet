
import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data definition 
st.title('🌨️Prévision météorologique🌨️')
st.write("La météorologie est une science qui a pour objet l'étude des phénomènes atmosphériques tels que les nuages, les précipitations ou le vent dans le but de comprendre comment ils se forment et évoluent en fonction des paramètres mesurés tels que la pression, la température et l'humidité.")
st.write("C'est une discipline qui traite principalement de la mécanique des fluides et de la thermodynamique mais qui fait usage de différentes autres branches de la physique, de la chimie et des mathématiques. Purement descriptive à l'origine, la météorologie est devenue un lieu d'application de ces disciplines.")
st.write("La météorologie moderne permet d'établir des prévisions de l'évolution du temps en s'appuyant sur des modèles mathématiques à court comme à long terme qui assimilent des données de nombreuses sources dont les stations, les satellites et les radars météorologiques. La météorologie a des applications dans des domaines très divers comme les besoins militaires, la production d'énergie, les transports (aériens, maritimes et terrestres), l'agriculture,  la construction, la photographie aérienne ou le cinéma. Elle est également appliquée pour la prévision de la qualité de l'air ou de plusieurs risques naturels d'origine atmosphérique.")

url="nmamadousow@gmail.com"
with st.sidebar:
    st.info('Data analysis and visualisation')
    st.markdown(''' 
    📧 [Openwather](<https://openweathermap.org/>)
    
    📧 [Météo Sénégal](<https://www.meteoart.com/africa/senegal?page=day#date=2023-07-18>)
    
    💻 [Code source](<nmamadousow@gmail.com>)
                ''')
    st.write('Made with 💗 by  ☎️ Contact',url)
    st.info('GOMYCODE SN')
    
## data exploratory analysis
st.title('💹Exploratory data analysis 💹')

df= pd.read_csv('weather.csv')
st.info('Data Frame')
st.write(df)

st.info('Heatmap isnull')
fig, ax = plt.subplots()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
st.pyplot(fig)

# Nous allons traiter les données manquents

df['Sunshine'].fillna(0, inplace=True)
df['WindGustDir'].fillna(0, inplace=True)
df['WindGustSpeed'].fillna(0, inplace=True)
df['WindSpeed9am'].fillna(0, inplace=True)
df['WindDir3pm'].fillna(0, inplace=True)
df['WindDir9am'].fillna(0, inplace=True)

st.info('Clean isnull')
fig, ax = plt.subplots()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
st.pyplot(fig)

st.info('Correlations graph')
fig, ax = plt.subplots()
sns.heatmap(df.corr(),annot=False,cbar=True)
st.pyplot(fig)

df.drop('RISK_MM', inplace=True,axis=1)
df.drop('WindGustDir', inplace=True,axis=1)
df.drop('WindDir9am', inplace=True,axis=1)
df.drop('WindDir3pm', inplace=True,axis=1)

# Nous utiliserons LabelEncoder pour gérer les étiquettes de nos principales prévisions
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['RainToday'] = LE.fit_transform(df['RainToday'])
df['RainTomorrow'] = LE.fit_transform(df['RainTomorrow'])

st.info('Date frame encoder')
st.write(df)

