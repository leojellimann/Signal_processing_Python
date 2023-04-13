# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:56:22 2021

@author: Léo
"""
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.signal import decimate, firwin
ecg = np.loadtxt('ecg_lfn.csv')
N=len(ecg)
Fe=1000
t=np.arange(N)/Fe#(valeur min,MAX,le pas)
plt.title("ecg")
plt.plot(t,ecg)

def tracer_TFD_et_TFD_inverse(signal,Te,t) :

    TFD = np.fft.fft( signal )
    k = np.arange( 0 , len(signal) )
    N = len(t)
    
    TFD_shifted = np.fft.fftshift( TFD )
    k_prime = k - np.floor( N / 2 )
    TFD_shifted_inv=np.fft.ifft(TFD_shifted)

    fe = 1 / Te
    f = k_prime * (fe / N)
    
    fig, axs = plt.subplots(3,figsize=(20, 8))
    fig.suptitle('TFD')
    axs[0].plot(f , np.abs( TFD_shifted ))
    axs[1].plot(f , np.angle( TFD_shifted ))
    axs[2].plot(t , TFD_shifted_inv)
   
    plt.grid()
    plt.show()

tracer_TFD_et_TFD_inverse(ecg,Fe,t)

#Création d'un filtre passe bas
ecg2 = decimate(ecg, 10)
plt.title("Signal ecg après application du filtre passe bas en fonction du temps")
plt.plot(ecg2)

#-------------------------------------------------------------------------------------
#Définition du gabarit
#-------------------------------------------------------------------------------------

#Gabarit passe haut
#période t vaut environ 7 sec donc la fréquence de coupure est 1/7
fc = 1/4
ftransition = 0.5
x= [0, 0.1, 0.1, 0.75, 1.25]
y= [-60, -60, 1, 1, 1]
x1= [0.35, 0.35, 0.75, 1.25]
y1= [-90, -1, -1, -1]

plt.figure(figsize=(5,3))
plt.plot(x, y, x1, y1, color='k', linestyle = '--', marker='*')
plt.title("Gabarit numérique")
plt.show()

#log10
#abs
#angle

#-------------------------------------------------------------------------------------
#Réalisation d'un filtre RIF
#-------------------------------------------------------------------------------------

#passe haut pour garder les pics avec fréquence de coupure de 1/6eme ou 1/7eme
#pour éliminer les basses fréquences
#Largeur de transition Blackman 5,5/N = (1,7)/fe
fe = 100
N = (5.5*fe)/ftransition#N = nombre de zéros
#fc*1,5 permet de décaler le filtre passe bas qu'on applique sur le ecg pour supprimer l'ondulation
coeffnum = firwin(N+1, fc*1.3, window=('kaiser', 6.7), pass_zero='highpass', nyq = 50)#synthétiser le filtre RIF
w,freq = sig.freqz(coeffnum, worN= 4096, fs=100)#fs = 100 verifier que le signal correspond bien au gabarit
#freqz traçe la réponse en fréquence

module = np.abs(freq)
gain = 20*np.log10(module)
plt.plot(w,gain)
plt.xlim(0,1)#valeurs axe abscisse en hertz
plt.ylabel("dB")
plt.xlabel("Hz")
plt.plot(x, y, x1, y1, color='k', linestyle = '--', marker='*')
plt.title("Filtre RIF par rapport au gabarit")
plt.show()

#Afficher le retard de groupe
w1, gd = sig.group_delay((coeffnum,1), fs=fe)#coeffnum = numérateur, 1 = dénominateur car en RIF pas de dénominateur
plt.plot(w1, gd)
plt.xlim(0,1)
plt.title("Retard de groupe du filtre RIF")
plt.show()#retard de groupe constant car on a un RIF, les deux pics au début sont des erreurs numériques

#afficher les poles et les zeros
z, p, k = sig.tf2zpk(coeffnum,1)
plt.plot(np.real(z), np.imag(z), "o", mfc='none')#affiche les zeros
plt.show()



#------------------------------------------------------------------------------------------
#Réalisation d'un filtre RII
#------------------------------------------------------------------------------------------


#Transformation des valeurs fréquences du filtre RIF pour réaliser le gabarit RII
fx1 = (fe/math.pi)*math.tan(math.pi*(0.1/fe))#0.75 = fréquence du point du gabarit à modifier pour adapter à l'analogique
fx2 = (fe/math.pi)*math.tan(math.pi*(0.75/fe))
fx3 = (fe/math.pi)*math.tan(math.pi*(0.35/fe))
#à mettre dans le x pour le nouveau gabarit


#Réalisation du gabarit du filtre analogique
x= [0, fx1, fx1, fx2, 1]#abscisse en fréquence
y= [-60, -60, 1, 1, 1]
x1= [fx3, fx3, fx2, 1]
y1= [-90, -1, -1, -1]

plt.figure(figsize=(5,3))
plt.plot(x, y, x1, y1, color='k', linestyle = '--', marker='*')
plt.title("Gabarit analogique")
plt.show()
#iirdesign passe filtre analogique en numérique
b,a = sig.iirdesign(wp=0.35, ws=0.1, gpass=1, gstop=60, ftype='cheby2', fs=fe)#cheby2 car il y a des ondulations en bande atténuée
w,h = sig.freqz(b=b, a=a, fs=fe, worN= 4096,)
gain=20*np.log10(np.abs(h+1e-12))
plt.plot(w, gain)
plt.xlim(0,1)
plt.ylabel("dB")
plt.xlabel("Hz")
plt.plot(x, y, x1, y1, color='k', linestyle = '--', marker='*')
plt.title("Réponse fréquentielle du filtre RII par rapport au gabarit")
plt.show()

w,gd = sig.group_delay((b,a))
plt.plot(w,gd)
plt.xlim(0,1)
plt.title("Retard de groupe du filtre RII")
plt.show()

#Calculer les poles et zéros
z,p,k = sig.tf2zpk(b,a)
teta = np.arange(0,2*math.pi, 0.01)#Dessine un cercle unitaire

plt.plot(np.real(z), np.imag(z), "o", mfc='none')#affiche les zeros
plt.plot(np.real(p), np.imag(p), "+", mfc='none')#affiche les poles
plt.plot(np.cos(teta), np.sin(teta), mfc='none')#affiche le cercle unitaire
plt.xlim(-1.05, 1.05)
plt.ylim(-1.05,1.05)
plt.show()

#------------------------------------------------------------------------------------
#Comparaison des filtres synthétisés
#------------------------------------------------------------------------------------

#pad permet de rajouter des 0 à la fin du signal pour pouvoir recentrer le reste du signal dans la fenetre
signal_adpt = np.pad(ecg2,(1,600))
srif = sig.lfilter(coeffnum, 1, x=signal_adpt, zi=None)
t2 = np.arange(0, len(srif))/100
plt.plot(t2,srif)
plt.xlabel("Sec")
plt.ylabel("ecg(t)")
plt.title("ECG après application du filtre RIF")
plt.show()#retard de groupe très long du RIF mais constant
#RIF plus robuste à la troncature

srii = sig.lfilter(b,a, x=ecg2, zi=None)
t2 = np.arange(0, len(srii))/100
plt.plot(t2,srii)
plt.xlabel("Sec")
plt.ylabel("ecg(t)")
plt.title("ECG après application du filtre RII")
plt.show()#Retard de groupe n'est pas forcément constant
#RII ordre fonction transfert beaucoup plus faible que RIF donc moins de coefficients à stocker


