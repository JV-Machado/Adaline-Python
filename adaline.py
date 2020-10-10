import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Adaline:
    
    def __init__(self, input_values, output_values, learning_rate, activation_function, accuracy):
        ones_column = np.ones((len(input_values), 1)) * -1
        self.input_values = np.append(ones_column, input_values, axis = 1)
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(self.input_values.shape[1])
        self.W0 = self.W
        self.accuracy = accuracy
        self.EqmPlot = []
        
    def train(self):
        epochs = 0
        Eqm1 = 0
        Eqm2 = 0
        
        while True:
            Eqm1 = self.EQM()
            
            print(f'Epochs: {epochs}')
            for x,d in zip(self.input_values, self.output_values):                
                u = np.dot(x, self.W)              
                self.W = self.W + self.learning_rate * (d - u) * x                
                y = self.activation_function.g(u)
                
                print(f'Input: {x} Output: {y} Expeted: {d}')
                
            epochs += 1
            Eqm2 = self.EQM()  
            EqmE = abs(Eqm2 - Eqm1)
            self.Plot(Eqm1, epochs, EqmE)
            if(EqmE <= self.accuracy):                   
                break
  
            print('')
        print('')  
        print(f'Final W: {self.W}')      
        
                
    def EQM(self):
        p = len(pd.read_csv('DataBase/treinamento.csv'))
        eqm = 0        
        for x,d in zip(self.input_values, self.output_values):  
            u = np.dot(x, self.W) 
            eqm = eqm + ((d - u)**2)
        eqm = eqm/p    
        return eqm
        
    def evaluate(self, Input_Teste):
        u = np.dot(Input_Teste, self.W)    
        return self.activation_function.g(u)

    def Plot(self, eqm1, epochs, eqme):    
        self.EqmPlot.append(eqm1)   
        x = np.arange(epochs)
        
        if(eqme <= self.accuracy):
            plt.plot(x, self.EqmPlot)        
            plt.show()
          