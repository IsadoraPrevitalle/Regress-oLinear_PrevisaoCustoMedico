import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#Lendo a base com a função read_csv do pandas
#Separando os valores pelas respectivas colunas com a função iloc do pandas
base = pd.read_csv('plano_saude.csv')
print(base)
X_plano = base.iloc[:,0].values #Idade - classe meta
Y_custo = base.iloc[:,1].values #custo do plano - previsor

#Trata-se os dados da base para treinar o modelo, transformando cada elemento em um vetor dentro de outro vetor
X_plano = X_plano.reshape(-1,1)

#Declara a função de regressão linear
#Treina o modelo na função linear com os dados de treino (por meio da função fit)
#Função fit minimiza os erros entre as variaveis para realizar a previsão, encontra o b0 e os coeficientes para o atributo
regressao_plano = LinearRegression()
regressao_plano.fit(X_plano, Y_custo)

#Gera as previsões por meio da formula y = b0 + b1 * x1, utilizando como base as idades fornecidas na base de dados
previsoes = regressao_plano.predict(X_plano)


#Testando a equação linear (b0+b1*40(idade)) de duas maneiras, com uma idade de 40 anos (fora da base)
print("Prevendo o custo para uma idade de 40 anos por meio do b0(intercept_) e b1(coef_): \nCusto do plano de saúde: ",regressao_plano.intercept_+regressao_plano.coef_*40)
print("Prevendo o custo para uma idade de 40 anos por meio da formula de previsão do modelo: \nCusto do plano de saúde: ",regressao_plano.predict([[40]]))

#verificando o coeficiente de determinação - métrica que indica o quão bem o modelo se ajusta
print("Modelo tem R^2: ",regressao_plano.score(X_plano, Y_custo))

#Calculando o Coeficiente de correlação com a função corrcoef do numpy
#print("variaveis tem um coeficiente de correlação de: ",np.corrcoef(X_plano, Y_custo))

#b0 e b1
print("Coeficiente de interceção: ",regressao_plano.intercept_)
print("coeficiente das variáveis independentes: ",regressao_plano.coef_)

#Idades e previsões dos custos do plano de saúde
print("Idade: ",X_plano.ravel())
print("Custo do plano de saúde: ",previsoes)

#Transformando os valores para serem plotados 
#Plota-se os valores reais, da base e os valores previstos por meio da regressão linear
grafico = px.scatter(x = X_plano.ravel(), y = Y_custo)
grafico.add_scatter(x=X_plano.ravel(), y=previsoes, name = "Regressão")
grafico.show()
