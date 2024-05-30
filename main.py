import streamlit as st 
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
scaler = StandardScaler()
label_encoder = LabelEncoder()


column_to_let = ["category","subcategory","subsubcategory", "is_view","is_cart","is_purchase","p_views","p_carts","p_purchases","user_views","user_carts","user_purchases"]


data = pd.read_csv("C:/Users/djibr/Desktop/Sc/Others/Ac/archive/datasetreco.csv", sep=";")
df= pd.read_csv("C:/Users/djibr/Desktop/Sc/Others/Ac/archive/datasetreco.csv", sep=";")[column_to_let]


df.drop_duplicates()


df['category'] = label_encoder.fit_transform(df['category'])
df['subcategory'] = label_encoder.fit_transform(df['subcategory'])
df['subsubcategory'] = label_encoder.fit_transform(df['subsubcategory'])


data['category'] = label_encoder.fit_transform(data['category'])
data['subcategory'] = label_encoder.fit_transform(data['subcategory'])
data['subsubcategory'] = label_encoder.fit_transform(data['subsubcategory'])


Y = df["is_purchase"]
X = df.drop(["is_purchase"], axis=1)
X_scaled = scaler.fit_transform(X)
df = pd.DataFrame(X_scaled, columns=X.columns)
df['is_purchase'] = Y


x_train, x_test, y_train, y_test= train_test_split(X,Y, test_size=0.2, random_state=3)

under_sampler = RandomUnderSampler(sampling_strategy= "not minority")


x_train_resampled, y_train_resampled = under_sampler.fit_resample(x_train, y_train)
x_test_resample, y_test_resample = under_sampler.fit_resample(x_test, y_test)


log_regression= LogisticRegression()
log_regression.fit(x_train_resampled, y_train_resampled)


# La probabilité
y_prob = log_regression.predict_proba(x_test_resample)[:, 1]


fpr, tpr, thresholds = roc_curve(y_test_resample, y_prob)


# Choix du seuil le plus élevé
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]


y_prod_optimal = (y_prob >= optimal_threshold).astype(int)


# Matrice de confusion
matrice_conf= metrics.confusion_matrix(y_test_resample, y_prod_optimal)


# Accuracy
print("accuracy : ",metrics.accuracy_score(y_test_resample, y_prod_optimal))


# Recall
print("recall: ",metrics.recall_score(y_test_resample, y_prod_optimal))


# ROC
y_pred_proba = log_regression.predict_proba(x_test_resample)[::,1]
fpr, tpr, optimal_threshold = metrics.roc_curve(y_test_resample,  y_prod_optimal)
auc = metrics.roc_auc_score(y_test_resample, y_prod_optimal)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.show()


def prediction(user_id, product_id, log_regression, data):
    x_prod = data[(data['user_id'] == int(user_id)) & (data['product_id'] == int(product_id))]
    
    # Sélectionner les colonnes caractéristiques
    X_user = x_prod
    X_user = X_user.drop(["Date","Time","event_type","user_id","product_id","price","is_purchase","user_session"],axis=1)
    
    # Prédire la probabilité
    proba = log_regression.predict_proba(X_user)[:, 1]
    
    return proba[0]


st.title("La probabilité pour qu'un utilisateur achète un produit donné")

st.sidebar.header('Entrer les informations utilisateur')
user_id = st.sidebar.number_input('user_id', min_value=1, max_value=100, value=1)
product_id = st.sidebar.number_input('product_id', min_value=1, max_value=1000, value=1)

if st.sidebar.button('Prédire'):
    try:
        proba = prediction(user_id, product_id, log_regression, data)
        st.write(f'la probalbilité que {user_id} achète le produit {product_id} est de : {proba:.2f}')
    except ValueError as e:
        st.write(str(e))

# Affichage de quelques données de l'ensemble d'entraînement
st.write("Extrait de l'ensemble de données :")
st.write(data.head())
