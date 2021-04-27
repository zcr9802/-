'''
Author:jordan
Describe:A file contains many regression method, and compare them with the existed ones.
Specific:lasso,
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from scipy import stats

def data_generator():
    x = np.random.random(10)
    for i in range(199):
        # for j in range(10):
        # x = np.row_stack((x, np.random.normal(0, 1, 10)))
        x = np.row_stack((x,np.random.randint(0,10,10)))
    weight = np.array([[10],[-9],[-11],[12],[0],[0],[0],[0],[9.5],[-10.5]])
    x = preprocessing.scale(x)/(200**0.5)
    print(x.shape)
    y = np.dot(x,weight)\
        +np.resize(np.random.normal(0,0.1,200),(200,1))
    # y = preprocessing.scale(y) / (200 ** 0.5)
    return x,y


def score(y_predict, y_true):
    u = (((y_true - y_predict)) ** 2).sum()
    v = (((y_true - y_true.mean())) ** 2).sum()
    r_2 = (1 - u / v)
    return r_2

class regression_lasso(object):
    weight = np.array([])
    a = 0
    def fit_by_coordinate_descent(self, independent_variable:np.array, dependent_variable:np.array, lambda_norm=0.5, iteration=500) -> np.array:
        '''
        The function to train the parameters of the model
        :param independent_variable: x, required to be a array(n*p)
        :param dependent_variable: y, required to be a array(n*1)
        :param lambda_norm: the lambda parameter
        :param iteration: the number of iterations
        :return: weight
        '''
        x = independent_variable
        n = x.shape[0]
        p = x.shape[1]
        y = dependent_variable
        # self.a = y.mean()
        self.weight = np.array(np.random.uniform(-1,1,p))
        self.weight_get_iteration_by_coordinate_descent(x,y,lambda_norm,iteration)
        return self.weight

    def fit_by_Stagewise(self, independent_variable:np.array, dependent_variable:np.array, lambda_norm=0.5, iteration=10000) -> np.array:
        '''
        The function to train the parameters of the model by using Least Angle Regression
        :param independent_variable: x, required to be a array(n*p)
        :param dependent_variable: y, required to be a array(n*1)
        :param lambda_norm: the lambda parameter
        :param iteration: the number of iterations
        :return: weight
        '''
        x = independent_variable
        n = x.shape[0]
        p = x.shape[1]
        y = dependent_variable
        # self.a = y.mean()
        self.weight = np.zeros(p)
        self.weight_get_iteration_by_Stagewise(x,y,lambda_norm,iteration)
        return self.weight

    def fit_by_LARS(self, independent_variable: np.array, dependent_variable: np.array, lambda_norm=0.5,
                         iteration=1) -> np.array:
        '''
        The function to train the parameters of the model by using Least Angle Regression
        :param independent_variable: x, required to be a array(n*p)
        :param dependent_variable: y, required to be a array(n*1)
        :param lambda_norm: the lambda parameter
        :param iteration: the number of iterations
        :return: weight
        '''
        x = independent_variable
        n = x.shape[0]
        p = x.shape[1]
        y = dependent_variable
        # self.a = y.mean()
        self.weight = np.zeros(p)
        self.weight_get_iteration_by_LARS(x, y, lambda_norm)
        return self.weight

    def weight_get_iteration_by_coordinate_descent(self, x:np.array, y, lambda_norm:float, iteration:int ,loss_abs =0.1 ):

        loss=0
        for i in tqdm(range(iteration)):
            for j in range(self.weight.reshape((-1, 1)).shape[0]):
                # print(np.dot(x[:, i].T , x[:, i]))
                a = (np.dot(x[:, j].T , x[:, j]))
                dw = np.array(np.zeros((self.weight.reshape((-1, 1)).shape[0], 1)))
                # print(dw)
                # print(self.weight)
                dw[j, 0] = self.weight.reshape((-1, 1))[j,0]
                c = -2 * np.dot(x.T ,(y - np.dot(x , (self.weight.reshape((-1, 1)) - dw))))[j, 0]
                if c <= -lambda_norm:
                    self.weight[j] = (-lambda_norm - c) / a / 2
                elif c >= lambda_norm:
                    self.weight[j] = (lambda_norm - c) / a / 2
                else:
                    self.weight[j] = 0
            # print(w)
            loss_new = np.dot((y-np.dot(x,self.weight.reshape((-1, 1)))).T,
                              (y-np.dot(x,self.weight.reshape((-1, 1)))))+\
                       lambda_norm*sum([abs(self.weight.reshape((-1, 1))[i,0])
                                        for i in range(self.weight.reshape((-1, 1)).shape[0])])
            if abs(loss-loss_new)<loss_abs:
                print('end of iteration')
                print('the number of iteration=',i)
                break
            loss=loss_new

    def weight_get_iteration_by_Stagewise(self, x:np.array, y, lambda_norm:float, iteration=10000 ,loss_abs=0.1):
        small_constant = 0.01
        for i in tqdm(range(iteration)):
            residual = y-np.dot(x,self.weight.reshape(-1,1))
            correlations = np.dot(x.T,residual)
            index_most_related_x = abs(correlations).argmax()
            self.weight[index_most_related_x] += np.sign(correlations[index_most_related_x])*small_constant
            if (sum(abs(y-np.dot(x,self.weight.reshape(-1,1)))))<=loss_abs:
                print(abs(y-np.dot(x,self.weight.reshape(-1,1))))
                print(i)
                break

    def weight_get_iteration_by_LARS(self, x: np.array, y, lambda_norm: float, iteration=100,
                                                   loss_abs=0.1):

        # print(x)
        #store the index of the most correlated x and the sign of the correlation of x and residual
        for k in range(iteration):
            most_correlated_set = []
            for i in tqdm(range(x.shape[1])):
                residual = y-np.dot(x,self.weight.reshape(-1,1))
                # print(sum(residual))
                # print(sum(residual))
                correlations = np.dot(x.T,residual)
                # print(correlations)
                correlations_not_set_A = np.array(correlations)
                print(correlations)
                print(correlations_not_set_A)
                # print(correlations_not_set_A)
                for j in most_correlated_set:
                    correlations_not_set_A[abs(j)-1,0] = 0
                # print(correlations_not_set_A)
                c_max = abs(correlations.max())
                # print(c_max)
                # print(correlations)
                index_most_related_x = abs(correlations_not_set_A).argmax()
                # print(correlations_not_set_A)
                # print(most_correlated_set)
                correlations_not_set_A[index_most_related_x,0]=0
                # print(correlations_not_set_A)
                most_correlated_set.append(int(np.sign(correlations[index_most_related_x])*(index_most_related_x+1)))
                # print(most_correlated_set)
                matrix_x_in_setA = np.array(list(map(lambda i:np.sign(i)*x[:,(abs(i)-1)],most_correlated_set))).T
                # print('compare')
                # print(matrix_x_in_setA)
                # print(1)
                # print(x)
                # print(2)
                # print(np.dot(x.T,x))
                # print(3)
                # print(np.dot(matrix_x_in_setA.T,matrix_x_in_setA))
                # print(x)
                # print(most_correlated_set)
                # print(matrix_x_in_setA)
                # print('adasdsa')
                # print(matrix_x_in_setA[:,-1])
                # print(np.dot(matrix_x_in_setA[:,-1].reshape(1,-1),matrix_x_in_setA[:,-1].reshape(-1,1)))
                x_A_T_x_A =np.dot(matrix_x_in_setA.T,matrix_x_in_setA)
                # print('wdadasdk')
                # print(x_A_T_x_A)
                constantofprojection_from_x_in_setA_to_equiangular_vector = np.dot(np.dot(np.ones(len(most_correlated_set)).reshape(1,-1),
                                                                                   np.linalg.inv(x_A_T_x_A)),np.ones(len(most_correlated_set)).reshape(-1,1))[0][0]**(-0.5)
                # print(constantofprojection_from_x_in_setA_to_equiangular_vector)
                equiagular_vector = np.dot(matrix_x_in_setA,constantofprojection_from_x_in_setA_to_equiangular_vector*\
                                    np.dot(np.linalg.inv(x_A_T_x_A),np.ones(len(most_correlated_set)))).reshape(-1,1)

                index_most_related_x_next = abs(correlations_not_set_A).argmax()
                print(correlations_not_set_A)
                print(index_most_related_x_next)
                # if constantofprojection_from_x_in_setA_to_equiangular_vector-np.dot(x[:, index_most_related_x_next].reshape(1, -1), equiagular_vector <0):
                #     print('wwww')
                l = [(c_max-correlations[index_most_related_x_next])/(constantofprojection_from_x_in_setA_to_equiangular_vector
                                                                      -np.dot(x[:,index_most_related_x_next].reshape(1,-1),equiagular_vector)),
                     (c_max + correlations[index_most_related_x_next]) / (
                                 constantofprojection_from_x_in_setA_to_equiangular_vector
                                 + np.dot(x[:, index_most_related_x_next].reshape(1, -1), equiagular_vector)),
                     ]
                # print(l)
                # for i in range(len(l)):
                #     if l[i]<0:
                #         l[i]=sum([abs(i) for i in l])
                #         print(l)
                # if len(l)==0:
                #     print('can not get the len of move')
                ls=[]
                for i in l:
                    ls.append(i)
                for i in range(len(l)):
                    if i<0:
                        ls.remove(i)
                if len(ls)==0:
                    print('error a')
                    return
                lengthofmove_equiangular_vector = min([abs(i) for i in ls])[0][0]
                print(c_max)
                print(correlations[index_most_related_x_next])
                # if correlations[index_most_related_x_next]>0:
                #     lengthofmove_equiangular_vector = l[0]
                #     print('dayu 0')
                # else:
                #     lengthofmove_equiangular_vector = l[1]
                # print('xt u')
                # print(np.dot(matrix_x_in_setA.T,matrix_x_in_setA))
                for i in most_correlated_set:
                    self.weight[abs(i)-1] += np.sign(i)*lengthofmove_equiangular_vector*constantofprojection_from_x_in_setA_to_equiangular_vector
                # print(np.dot((y-np.dot(x,self.weight.reshape(-1,1))).T,x))
                # print(self.weight)
                print(self.weight)
                print(most_correlated_set)

    def fit(self,independent_variable:np.array, dependent_variable:np.array, type = 'lars',lambda_norm=0.5, iteration=500):
        '''

        :param independent_variable: np.array
        :param dependent_variable: np.array
        :param type: 'lars','lasso','forward.stagewise'
        :param lambda_norm:
        :param iteration:
        :return:
        '''
        if type == 'lasso':
            self.fit_by_coordinate_descent(independent_variable=independent_variable,
                                           dependent_variable=dependent_variable,lambda_norm=lambda_norm,iteration=iteration),
        elif type == 'lars':
            self.fit_by_LARS(independent_variable=independent_variable,
                             dependent_variable=dependent_variable,lambda_norm=lambda_norm,iteration=iteration)
        elif type == 'forward.stagewise':
            self.fit_by_Stagewise(independent_variable=independent_variable,
                                  dependent_variable=dependent_variable,lambda_norm=lambda_norm,iteration=iteration)
        else:
            print("input type is incorrect")
            print("choose the default type 'lars'")
            self.fit_by_LARS(independent_variable=independent_variable,
                             dependent_variable=dependent_variable,lambda_norm=lambda_norm,iteration=iteration)


    def predict(self, x:np.array):
        return np.dot(x,self.weight.T)

    def score(self, x:np.array, y) ->float:
        y_true = y
        y_predict = np.dot(x,self.weight.T)
        u = (((y_true-y_predict))**2).sum()
        v = (((y_true-y_true.mean()))**2).sum()
        r_2 = (1-u/v)
        return r_2

class regression_bayes(object):
    weight = np.array([])
    stats.norm.pdf(0.5,0,1)
    pass



def lasso():
    x,y = data_generator()
    # print(sum([i**2 for i in x[:,0]]))
    # print(x[:,0].reshape(1,-1))
    print(np.dot(x.T,x))
    lasso = regression_lasso()
    stagewise = regression_lasso()
    lars = regression_lasso()
    x_train = x
    y_train = y
    # x_train,x_test, y_train, y_test  = \
    # train_test_split(x,y,test_size=0.5, random_state=0)
    # print(np.dot(x_train.T, y_train))
    # print(x_train,y_train)
    # print(x_train)
    # print(load_breast_cancer()['target'].shape)
    lasso.fit_by_coordinate_descent(x_train,y_train,lambda_norm=3)
    stagewise.fit_by_Stagewise(x_train,y_train)
    lars.fit_by_LARS(x_train,y_train)
    # print(lasso.weight)
    # print(lasso.weight)
    # print(lasso.score(x_test,y_test))
    lasso_sklearn = Lasso(alpha=0.01)
    lasso_sklearn.fit(x_train, y_train)
    print('the ture beta is')
    print(np.array([[10],[-9],[-11],[12],[0],[0],[0],[0],[9.5],[-10.5]]).reshape((1,-1)))
    print('the beta calculate by function Lasso in sklearn is')
    print(lasso_sklearn.coef_)
    print('the beta calculate by function regression_lasso made by zcr is')
    print(lasso.weight)
    print('the beta calculate by function stagewise made by zcr is')
    print(stagewise.weight)
    print('the beta calculate by function lars made by zcr is')
    print(lars.weight)
    # print(x)
    # for i in range(x.shape[1]):
    #     print(x[:,i])
# y_predict = lasso_sklearn.predict(x_test)
# print(score(y_predict, y_test))
lasso()
# lasso.fit(train_data,train_targe)
