# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 08:59:53 2021

@author: Léo
"""

from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.signal import decimate, firwin


#Créez un vecteur contenant le motif h de durée T = 10^-7s et échantillonné à 10^9s
time = 1e-7
Te = 1e-9#nombre d'échantillons 
abscissesec = np.arange(0, time, Te)#Créer l'axe des abscisse sur lequel on affiche le signal
                        #début, fin, point par echantillon
h = np.where(abscissesec < (time/2), 1, -1)#Signal quand on veut envoyer un 1. Si < time/2 --> 1 sinon -1

plt.title("signal h en fonction du temps")
plt.plot(abscissesec,h)
plt.xlabel("Temps en seconde")
plt.ylabel("Valeur en binaire")
plt.show()

#------------------------------------------------------------------------------------

#Concaténation de plusieurs motifs pour obtenir la séquence 1 0 1 1 

T = time*4
RSB = 10
newabscissesec = np.arange(0, T, Te)
sequence = np.concatenate((h, -h, h, h), axis=0)
plt.title("Concaténation de plusieurs signaux h en fonction du temps")
plt.plot(newabscissesec, sequence)
plt.xlabel("Temps en seconde")
plt.ylabel("Valeur en binaire")

#---------------------------------------------------------------------------

#Ajout d'un bruit gaussien au signal
N = len(sequence)
bgaussien = np.random.normal(0, 1, N)#(moyenne, sigma, nombre de valeurs aléatoires qu'on veut
hbruit = sequence + bgaussien
plt.title("signal h avec du bruit en fonction du temps")
plt.plot(newabscissesec, hbruit)
plt.xlabel("Temps en seconde")
plt.ylabel("Valeur en binaire + bruit")
plt.show()

#----------------------------------------------------------

#Ajout d'un bruit gaussien adapté aux paramètres RSB=10
Px = np.linalg.norm(hbruit**2)/N#Puissance du signal
sigma = np.sqrt(Px)*math.pow(10,(-RSB/20))#Calcul du signal

bgaussien2 = np.random.normal(0, sigma, N)
hbruit2 = sequence + bgaussien2#Réalisation du signal bruité avec la sequence
plt.title("signal h avec du bruit et sigma calculé en fonction du RSB,  en fonction du temps")
plt.plot(newabscissesec, hbruit2)
plt.xlabel("Temps en seconde")
plt.ylabel("Valeur en binaire + bruit")
plt.show()

matchedfilter = np.correlate(hbruit2, h, mode='same')#Comparaison entre signal et bruit pour trouver les ressemblances
                            #h est la "réponse impulsionnelle" ici
plt.plot(newabscissesec, matchedfilter)
plt.axvline(x=0*(1e-7))
plt.axvline(x=1*(1e-7))
plt.axvline(x=2*(1e-7))
plt.axvline(x=3*(1e-7))
plt.axvline(x=4*(1e-7))

plt.title("Pics montrant les ressemblances entre le signal bruité et le motif h original")
plt.xlabel("Temps en seconde")
plt.ylabel("Valeur en binaire")
plt.show()

manchestersignal = np.loadtxt('manchester.csv')
Nm = len(manchestersignal)#Nombre d'échantillons
#print(Nm)
Tm = np.arange(0, Nm*Te, Te)#Permet de faire l'axe des abscisses en secondes (Nm*Te -> secondes)
#Tm = np.arange(0, Nm, 1)#Permet de faire l'axe des abscisses en échantillon, 1 échantillon par sec
#print(Tm)

corrmanch = np.correlate(manchestersignal, h, mode='same')
plt.plot(Tm, corrmanch)
plt.show()

#matchedmanchester = np.correlate(hbruit2, manchestersignal, mode='same')

#-----------------------------------------------------------------------------------
#Réalisation d'un filtre moyenneur sur les données d'un CSV
annee, temp = np.loadtxt("temperatures.csv", delimiter=' ', usecols=(0, 1), unpack=True )
plt.title("Lecture des données du fichier temperatures.csv en année")
plt.xlabel("Valeurs de température")
plt.ylabel("En années")
plt.show()

M = 5
#pour faire le filtre moyenneur : 
#convolution entre signal et réponse impulsionnelle (avec comme réponse impulsionnelle une porte de la taille de M)
po = np.ones(M)
porte = po*1/M#Création de la porte 
smooth = np.convolve(temp, porte, 'same')
plt.title("Difference entre données des température avant et après mise en place du moyenneur")
plt.plot(annee, temp)#Affichage des données
plt.plot(annee, smooth)#Affichage des données moyennées
plt.xlabel("Valeurs de température")
plt.ylabel("En années")
plt.show()

#------------------------------------------------------------------------------------

#Approximation par moindres carrés

xannee = (annee-1880)/140
y = np.transpose(np.array([temp]))
H = np.transpose(np.array([ xannee**0, xannee**1, xannee**2 ]))

Ht = np.transpose(H)
HtH = np.dot(Ht, H)
HtHinv = np.linalg.pinv(HtH)
HtHinvHt = np.dot(HtHinv, Ht)
teta = np.dot(HtHinvHt, y)
mc = teta[0] + teta[1]*xannee + teta[2]*(xannee**2)
plt.grid()
plt.plot(xannee, temp)
plt.plot(xannee, mc)
plt.title("Réchauffement climatique")
plt.show()

#


