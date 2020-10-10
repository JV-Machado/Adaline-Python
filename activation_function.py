from abc import ABC, abstractmethod

class Activation_Function(ABC):
    @staticmethod
    @abstractmethod
    def g(u):
        pass    
    
class BinaryStep(Activation_Function):
    def g(u):
        return 1 if u >= 0 else 0
    
class SignFunction(Activation_Function):
    def g(u):
        return 1 if u >= 0 else -1
