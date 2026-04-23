import pandas as pd
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def initialize_model() -> Pipeline:
    #TODO
    model=Pipeline(
        [
            ("rbm",BernoulliRBM(random_state=42)),
            ("classfier",LogisticRegression(random_state=42))
        ]
    )
    return model
def read_and_train(model: Pipeline, file_path: str) -> Pipeline:
    #TODO
    data=pd.read_csv(file_path)
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    model.fit(x,y)
    return model

if __name__ == '__main__':

    file_path = "classification_data.csv"
    model = initialize_model()
    read_and_train(model, file_path)