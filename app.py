import pandas as pd

from activation_function import SignFunction

from adaline import Adaline

dataset = pd.read_csv('DataBase/treinamento.csv') 
X = dataset.iloc[:, 0:4].values
d = dataset.iloc[:, 4:].values
a = Adaline(X, d, 0.0025, SignFunction, 1e-6)

a.train()

print()
print(f'Initial W: {a.W0}')

print()
print(">>>>>>TESTE<<<<<<")
print(f'Input [0.9694,0.6909,0.4334,3.4965], Output {a.evaluate([-1, 0.9694,0.6909,0.4334,3.4965])}') 
print(f'Input [0.5427,1.3832,0.6390,4.0352], Output {a.evaluate([-1, 0.5427,1.3832,0.6390,4.0352])}') 
print(f'Input [0.6081,-0.9196,0.5925,0.1016], Output {a.evaluate([-1,  0.6081,-0.9196,0.5925,0.1016])}') 
print(f'Input [-0.1618,0.4694,0.2030,3.0117], Output {a.evaluate([-1,  -0.1618,0.4694,0.2030,3.0117])}') 
print(f'Input [0.1870,-0.2578,0.6124,1.7749], Output {a.evaluate([-1,  0.1870,-0.2578,0.6124,1.7749])}') 
print(f'Input [0.4891,-0.5276,0.4378,0.6439], Output {a.evaluate([-1, 0.4891,-0.5276,0.4378,0.6439])}') 
print(f'Input [0.3777,2.0149,0.7423,3.3932], Output {a.evaluate([-1,  0.3777,2.0149,0.7423,3.3932])}') 
print(f'Input [1.1498,-0.4067,0.2469,1.5866], Output {a.evaluate([-1, 1.1498,-0.4067,0.2469,1.5866])}')
print(f'Input [0.9325,1.0950,1.0359,3.3591], Output {a.evaluate([-1, 0.9325,1.0950,1.0359,3.3591])}') 
print(f'Input [0.5060,1.3317,0.9222,3.7174], Output {a.evaluate([-1, 0.5060,1.3317,0.9222,3.7174])}') 
print(f'Input [0.0497,-2.0656,0.6124,-0.6585], Output {a.evaluate([-1, 0.0497,-2.0656,0.6124,-0.6585])}') 
print(f'Input [0.4004,3.5369,0.9766,5.3532], Output {a.evaluate([-1, 0.4004,3.5369,0.9766,5.3532])}')
print(f'Input [-0.1874,1.3343,0.5374,3.2189], Output {a.evaluate([-1, -0.1874,1.3343,0.5374,3.2189])}')
print(f'Input [0.5060,1.3317,0.9222,3.7174], Output {a.evaluate([-1, 0.5060,1.3317,0.9222,3.7174])}')
print(f'Input [1.6375,-0.7911,0.7537,0.5515], Output {a.evaluate([-1, 1.6375,-0.7911,0.7537,0.5515])}')
print()







