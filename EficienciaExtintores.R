
###############################################################################
###############################################################################
###############################################################################

# Projeto Feedback 02
# Curso: Big Analytics com R e Microsoft Azure Machine Learning 3
#Formação Cientista de Dados 3.0
#Aluna: Fabiana Martins da Silva
#Contato: ms.fabiana@yahoo.com.br

#Objetivo: Prever o funcionamento/eficiência de extintores de incêndio com base
#em simulações feitas em computador e assim incluir uma camada adicional
#de segurança nas operações de uma empresa

###############################################################################
###############################################################################
###############################################################################

#Bibliotecas necessárias para o script
library(readxl)
library(dplyr)
library(ggplot2)
library(cowplot)
library(corrplot)
library(caret)
library(randomForest)
library(xgboost)
library(e1071)
library(pROC)
library(rpart.plot)

#Leitura do dataset
dados = read_excel("./Acoustic_Extinguisher_Fire_Dataset/Acoustic_Extinguisher_Fire_Dataset.xlsx")
View(dados)

##########################################################################
#Entendendo o dataset
##########################################################################

str(dados)
#Um total de 17442 testes foram realizados

unique(dados$FUEL)
#4 diferentes combustiveis

#5 tamanhos diferentes de latas de combustível liquido
#foram usados para atingir diferentes tamanhos de chamas
#Recorded as 7 cm=1, 12 cm=2, 14 cm=3, 16 cm=4, 20 cm=5
unique(dados$SIZE)

#Size igual a 6 ou 7 significa meio e cheio de gas que foi usado para combustível GLP
unique(dados%>%
  filter(SIZE == c(6, 7))%>%
  select(FUEL))

#Durante o experimento, o recipiente de combustível a 10cm de distância foi movido 
#para frente até 190cm aumentando 10 por vez
unique(dados$DISTANCE)

#foram 54 ondas sonoras de frequencias diferentes em cada distancia e tamanho de chama
unique(dados$FREQUENCY)
  
experimentos <- dados%>%
  select(FUEL, DISTANCE, SIZE, FREQUENCY)%>%
  group_by(FUEL, DISTANCE, SIZE)%>%
  summarise(n_frequencias = n_distinct(FREQUENCY))

#comprova que são 54 frequencias para cada configuração de experimento
#um total de 323 comfigurações
#323x54 totalizando 17442 testes foram realizados
unique(experimentos$n_frequencias)
dim(experimentos)

sum(is.na(dados)) #o dataset não possui valores NA

##########################################################################
#Explorando os dados
##########################################################################
str(dados)

table(dados$FUEL, dados$STATUS)
ggplot(dados)+
  geom_bar(mapping = aes(x = FUEL, fill= as.factor(STATUS)), position ='dodge')

table(dados$SIZE, dados$STATUS)
ggplot(dados)+
  geom_bar(mapping = aes(x = SIZE, fill= as.factor(STATUS)), position ='dodge')

#Pela correlação, percebemos que AIRFLOW está bastante correlacionado com 
#o sucesso dos experimentos, o que já era de se esperar
#Ele tem tb uma correlação alta com a distânciaq, o que tb é esperado
correl = cor(dados%>% select(-FUEL))
corrplot(correl, method = 'number')

#checando balanceamento de STATUS
prop.table(table(dados$STATUS))#possui um bom balanceamento entre os tipos

#Ajuste do tipo de algumas variáveis
dados$STATUS <- factor(dados$STATUS, levels = c(0, 1), labels = c("no", "yes"))
dados$FUEL <- as.factor(dados$FUEL)
dados$SIZE <- as.factor(dados$SIZE)

ggplot(dados, aes(x = STATUS, y = DISTANCE)) + 
  geom_boxplot() + 
  labs(title = "DISTANCE por STATUS")

ggplot(dados, aes(x = STATUS, y = DESIBEL)) + 
  geom_boxplot() + 
  labs(title = "DESIBEL por STATUS")

ggplot(dados, aes(x = STATUS, y = AIRFLOW)) + 
  geom_boxplot() + 
  labs(title = "AIRFLOW por STATUS")

ggplot(dados, aes(x = STATUS, y = FREQUENCY)) + 
  geom_boxplot() + 
  labs(title = "FREQUENCY por STATUS")

############################################################################
#Preparação modelagem
############################################################################
set.seed(42)

# Separação treino/teste
index <- createDataPartition(dados$STATUS, p = 0.7, list = FALSE)
treino <- dados[index, ]
teste <- dados[-index, ]

# Controle de validação cruzada
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  savePredictions = TRUE
)

############################################################################
#Modelagem Logistic Regression
############################################################################
print("###### Logistic Regression - Modelo 01 - Todas as variáveis ######")
modelo_glm1 <- train(
  STATUS ~ .,
  data = treino,
  method = "glm",
  family = "binomial",
  preProcess = c("center", "scale"),
  trControl = ctrl
)

print('Resultados de Treino')
modelo_glm1$results

pred_glm1 <- predict(modelo_glm1, newdata = teste)
confusionMatrix(pred_glm1, teste$STATUS)

probs_glm1 <- predict(modelo_glm1, newdata = teste, type = "prob")[, "yes"]

# Curva ROC
roc_glm1 <- roc(teste$STATUS, probs_glm1)
auc_glm1 = auc(roc_glm1)
auc_glm1

print("###### Logistic Regression - Modelo 02 - Sem a variável DISTANCE ######")
modelo_glm2 <- train(
  STATUS ~ . - DISTANCE,
  data = treino,
  method = "glm",
  family = "binomial",
  preProcess = c("center", "scale"),
  trControl = ctrl
)

print('Resultados de Treino')
modelo_glm2$results

pred_glm2 <- predict(modelo_glm2, newdata = teste)
confusionMatrix(pred_glm2, teste$STATUS)

probs_glm2 <- predict(modelo_glm2, newdata = teste, type = "prob")[, "yes"]

# Curva ROC
roc_glm2 <- roc(teste$STATUS, probs_glm2)
auc_glm2 = auc(roc_glm2)
auc_glm2


#Como pode ser observado abaixo, retirar DISTANCE que apresentou alta correlação
#com AIRFLOW não apresentou diferença relevante
ggroc(list(
  "GLM Modelo 01" = roc_glm1,
  "GLM Modelo 02" = roc_glm2),
  size = 1.5) +
  ggtitle("Comparação de Curvas ROC - GLM") +
  theme_minimal() +
  scale_color_manual(values = c("GLM Modelo 01" = "blue", "GLM Modelo 02" = "darkorange")) +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC Modelo 01 =", round(auc_glm1, 3),
                         "\nAUC Modelo 02 =", round(auc_glm2, 3)),
           size = 5, color = "black")


############################################################################
#Modelagem Decision Tree
############################################################################
print("###### Decision Tree ######")
modelo_dt <- train(
  STATUS ~ . - DISTANCE, #não usarei DISTANCE devido colinearidade 
  data = treino,
  method = "rpart",
  trControl = ctrl,
  tuneLength = 5  # Testa 5 valores de cp (complexidade da árvore)
)

print('Resultados de Treino')
modelo_dt$results
modelo_dt$bestTune

# Previsões
pred_dt <- predict(modelo_dt, newdata = teste)
confusionMatrix(pred_dt, teste$STATUS)

# Probabilidades da classe positiva
probs_dt <- predict(modelo_dt, newdata = teste, type = "prob")[, "yes"]

# Curva ROC
roc_dt <- roc(teste$STATUS, probs_dt)
auc_dt = auc(roc_dt)
auc_dt

ggroc(roc_dt, color = "purple", size = 1.5) +
  ggtitle("Curva ROC - Decision Tree") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc_dt, 3)),
           color = "purple", size = 5) +
          theme_minimal()

rpart.plot(modelo_dt$finalModel, main = "Árvore de Decisão")


############################################################################
#Modelagem Random Forest
############################################################################
print("###### Random Forest ######")
modelo_rf <- train(STATUS ~ . - DISTANCE, #não usarei DISTANCE devido colinearidade
                   data = treino, 
                   method = "rf", 
                   trControl = ctrl,
                   importance = TRUE, 
                   preProcess = c("center", "scale")
                   )

print('Resultados de Treino')
modelo_rf$results
modelo_rf$bestTune

pred_rf <- predict(modelo_rf, newdata = teste)
confusionMatrix(pred_rf, teste$STATUS)

varImpPlot(modelo_rf$finalModel)

probs_rf <- predict(modelo_rf, newdata = teste, type = "prob")[, "yes"]

# Curva ROC
roc_rf <- roc(teste$STATUS, probs_rf)
auc_rf = auc(roc_rf)
auc_rf

ggroc(roc_rf, color = "purple", size = 1.2) +
  ggtitle("Curva ROC - Random Forest") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc_rf, 3)),
           color = "purple", size = 5) +
  theme_minimal()

############################################################################
#Modelagem XGBoost
############################################################################
print("###### XGBoost ######")
modelo_xgb <- train(
  STATUS ~ . - DISTANCE, #não usarei DISTANCE devido colinearidade
                    data = treino,
                    method = "xgbTree",
                    trControl = ctrl, 
                    preProcess = c("center", "scale")
                    )

print('Resultados de Treino')
modelo_xgb$results
modelo_xgb$bestTune

pred_xgb <- predict(modelo_xgb, newdata = teste)
confusionMatrix(pred_xgb, teste$STATUS)

plot(varImp(modelo_xgb))

probs_xgb <- predict(modelo_xgb, newdata = teste, type = "prob")[, "yes"]

# Curva ROC
roc_xgb <- roc(teste$STATUS, probs_xgb)
auc_xgb = auc(roc_xgb)
auc_xgb

ggroc(roc_xgb, color = "purple", size = 1.2) +
  ggtitle("Curva ROC - XGBoost") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc_xgb, 3)),
           color = "purple", size = 5) +
  theme_minimal()

############################################################################
#Modelagem SVM com kernel radial
############################################################################
print("###### SVM (Radial) ######")
modelo_svm <- train(
                    STATUS ~ .,
                    data = treino,
                    method = "svmRadial",
                    trControl = ctrl, 
                    preProcess = c("center", "scale")
                    )


print('Resultados de Treino')
modelo_svm$results
modelo_svm$bestTune

pred_svm <- predict(modelo_svm, newdata = teste)
confusionMatrix(pred_svm, teste$STATUS)

probs_svm <- predict(modelo_svm, newdata = teste, type = "prob")[, "yes"]

# Curva ROC
roc_svm <- roc(teste$STATUS, probs_svm)
auc_svm = auc(roc_svm)
auc_svm

ggroc(roc_svm, color = "purple", size = 1.2) +
  ggtitle("Curva ROC - SVM (Radial)") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc_svm, 3)),
           color = "purple", size = 5) +
  theme_minimal()

############################################################################
#Modelagem K-Nearest Neighbors (KNN)
############################################################################
print("###### K-Nearest Neighbors (KNN) ######")
modelo_knn <- train(
  STATUS ~ . - DISTANCE,  # Removendo DISTANCE pela colinearidade com AIRFLOW
  data = treino,
  method = "knn",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneLength = 10  # Testa 10 valores diferentes de k
)

print('Resultados de Treino')
modelo_knn$results
modelo_knn$bestTune

pred_knn <- predict(modelo_knn, newdata = teste)
confusionMatrix(pred_knn, teste$STATUS)

probs_knn <- predict(modelo_knn, newdata = teste, type = "prob")[, "yes"]

# Curva ROC
roc_knn <- roc(teste$STATUS, probs_knn)
auc_knn = auc(roc_knn)
auc_knn

ggroc(roc_knn, color = "purple", size = 1.2) +
  ggtitle("Curva ROC - KNN") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc_knn, 3)),
           color = "purple", size = 5) +
  theme_minimal()

############################################################################
#Modelagem Naive Bayes
############################################################################
print("###### Naive Bayes ######")
modelo_nb <- train(
  STATUS ~ . - DISTANCE,  # Removendo DISTANCE pela colinearidade com AIRFLOW
  data = treino,
  method = "naive_bayes",
  trControl = ctrl,
  preProcess = c("center", "scale")
)

print('Resultados de Treino')
modelo_nb$results
modelo_nb$bestTune

pred_nb <- predict(modelo_nb, newdata = teste)
confusionMatrix(pred_nb, teste$STATUS)

probs_nb <- predict(modelo_nb, newdata = teste, type = "prob")[, "yes"]

# Curva ROC
roc_nb <- roc(teste$STATUS, probs_nb)
auc_nb = auc(roc_nb)
auc_nb

ggroc(roc_nb, color = "purple", size = 1.2) +
  ggtitle("Curva ROC - Naive Bayes") +
  annotate("text", x = 0.6, y = 0.1,
           label = paste("AUC =", round(auc_nb, 3)),
           color = "purple", size = 5) +
  theme_minimal()


############################################################################
# Comparação
############################################################################
# Criação da tabela com as métricas (substitua os valores com os reais do seu output)
comparativo_modelos <- data.frame(
  Modelo = c("GLM Modelo 01", "GLM Modelo 02", "Decision Tree", 
             "Random Forest", "XGBoost", "SVM (Radial)", "KNN", "Naive Bayes"),
  Acuracia = c(
    confusionMatrix(pred_glm1, teste$STATUS)$overall["Accuracy"],
    confusionMatrix(pred_glm2, teste$STATUS)$overall["Accuracy"],
    confusionMatrix(pred_dt,   teste$STATUS)$overall["Accuracy"],
    confusionMatrix(pred_rf,   teste$STATUS)$overall["Accuracy"],
    confusionMatrix(pred_xgb,  teste$STATUS)$overall["Accuracy"],
    confusionMatrix(pred_svm,  teste$STATUS)$overall["Accuracy"],
    confusionMatrix(pred_knn,  teste$STATUS)$overall["Accuracy"],
    confusionMatrix(pred_nb,   teste$STATUS)$overall["Accuracy"]
  ),
  Kappa = c(
    confusionMatrix(pred_glm1, teste$STATUS)$overall["Kappa"],
    confusionMatrix(pred_glm2, teste$STATUS)$overall["Kappa"],
    confusionMatrix(pred_dt,   teste$STATUS)$overall["Kappa"],
    confusionMatrix(pred_rf,   teste$STATUS)$overall["Kappa"],
    confusionMatrix(pred_xgb,  teste$STATUS)$overall["Kappa"],
    confusionMatrix(pred_svm,  teste$STATUS)$overall["Kappa"],
    confusionMatrix(pred_knn,  teste$STATUS)$overall["Kappa"],
    confusionMatrix(pred_nb,   teste$STATUS)$overall["Kappa"]
  ),
  AUC = c(
    auc_glm1, auc_glm2, auc_dt, auc_rf,
    auc_xgb, auc_svm, auc_knn, auc_nb
  )
)

# Visualiza a tabela ordenada por AUC
comparativo_modelos <- comparativo_modelos[order(-comparativo_modelos$AUC), ]
View(comparativo_modelos)


ggplot(comparativo_modelos, aes(x = reorder(Modelo, AUC), y = AUC)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Comparação de AUC entre Modelos",
       x = "Modelo",
       y = "AUC") +
  theme_minimal(base_size = 13)













