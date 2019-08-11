import numpy as np 

x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5,0.5], [2,0.5], [5.5,1], [1,1], [4,1.5]), dtype=float) # données d'entrer
y = np.array(([1], [0], [1],[0],[1],[0],[1],[0]), dtype=float) # données de sortie /  1 = rouge /  0 = bleu

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
x_entrer = x_entrer/np.amax(x_entrer, axis=0) # On divise chaque entré par la valeur max des entrées

# On récupère ce qu'il nous intéresse
X = np.split(x_entrer, [8])[0] # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice
xPrediction = np.split(x_entrer, [8])[1] # Valeur que l'on veut trouver

#Notre classe de réseau neuronal
class Neural_Network(object):
  def __init__(self):
        
  #Nos paramètres
    self.inputSize = 2 # Nombre de neurones d'entrer
    self.outputSize = 1 # Nombre de neurones de sortie
    self.hiddenSize = 3 # Nombre de neurones cachés

  #Nos poids
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) Matrice de poids entre les neurones d'entrer et cachés
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) Matrice de poids entre les neurones cachés et sortie


  #Fonction de propagation avant
  def forward(self, X):

    self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les poids W1
    self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)
    self.z3 = np.dot(self.z2, self.W2) # Multiplication matricielle entre les valeurs cachés et les poids W2
    o = self.sigmoid(self.z3) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
    return o

  # Fonction d'activation
  def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  # Dérivée de la fonction d'activation
  def sigmoidPrime(self, s):
    return s * (1 - s)

  #Fonction de rétropropagation
  def backward(self, X, y, o):

    self.o_error = y - o # Calcul de l'erreur
    self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur

    self.z2_error = self.o_delta.dot(self.W2.T) # Calcul de l'erreur de nos neurones cachés 
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # Application de la dérivée de la sigmoid à cette erreur

    self.W1 += X.T.dot(self.z2_delta) # On ajuste nos poids W1
    self.W2 += self.z2.T.dot(self.o_delta) # On ajuste nos poids W2

  #Fonction d'entrainement 
  def train(self, X, y):
        
    o = self.forward(X)
    self.backward(X, y, o)

  #Fonction de prédiction
  def predict(self):
        
    print("Donnée prédite apres entrainement: ")
    print("Entrée : \n" + str(xPrediction))
    print("Sortie : \n" + str(self.forward(xPrediction)))

    if(self.forward(xPrediction) < 0.5):
        print("La fleur est BLEU ! \n")
    else:
        print("La fleur est ROUGE ! \n")


NN = Neural_Network()

for i in range(1000): #Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    print("# " + str(i) + "\n")
    print("Valeurs d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(y))
    print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)

NN.predict()
