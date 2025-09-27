from numpy import np
class Perceptron:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weight=None
        self.bias=None
        self.acitvation_func=unit_step_func
        
    def fit(self,x,y):
        y_=np.array([ 1 if i > 0 else 0 for i in y])
        n_samples,n_features=x.shape
        self.weight=np.zeros(n_samples)
        self.bias=0
        for _ in self.n_iters:
            for index,value in enumerate(x):
                z=np.dot(self.weight.T,x)+self.bias
                ypred=self.acitvation_func(z)
                update=(y_[index]-ypred[index])*self.lr
                self.weight+=update*value
                self.bias+=update

        
    def predict(self,x):
         z=np.dot(self.weight.T,x)+self.bias
         ypred=self.acitvation_func(z)
         return ypred
    def unit_step_func(self,x):
        return np.where(x>=0,1,0)
