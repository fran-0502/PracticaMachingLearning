import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("juegos-ml.csv")
x = df.drop(columns=("juegos"))
y = df["juegos"]
x_entrenar, x_prueba, y_entrenar, y_pruebas = train_test_split(x,y,test_size=0.2)


modelo = DecisionTreeClassifier()
modelo.fit(x_entrenar,y_entrenar)
predicciones = modelo.predict(x_prueba)
 
puntaje = accuracy_score(y_pruebas, predicciones)
puntaje
 