import numpy as np


class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        z=1/(1+np.exp(-x))
        return z

    def cost_fun(self, theta,X,y,lambda_=0):
        m = y.size  # number of training examples
        n= theta.size
        grad = np.zeros(theta.shape)
        # for i in range(m):
        #     J+=-y[i]*np.log(self.sigmoid(np.dot(X[i],theta)))/m-(1-y[i])*np.log(1-self.sigmoid(np.dot(X[i],theta)))/m
        J=0
        J+=-np.dot(y,np.log(self.sigmoid(np.dot(X.T,theta))))/m-np.dot(1-y,np.log(1-self.sigmoid(np.dot(X.T,theta))))/m
        grad+=np.dot(X,(self.sigmoid(np.dot(X.T,theta))-y))/m
        J+=np.dot(theta,theta)*lambda_/2
        grad+=theta*lambda_
        return J, grad

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e5,lambda_=0):

        # number of training examples
        n,m= X.shape
        theta=np.random.randn(n)
        J_his=[]
        for i in range(int(max_iter)):
            theta_0=theta.copy()
            J,grad=self.cost_fun(theta_0,X,y,lambda_)
            theta-=lr*grad
            J_his.append(J)
            if (abs(J-J_his[i-1])<tol )&(i>=1000):
                return theta, J_his
        return theta,J_his

    def predict(self, X,y,theta):
        thereshold=0.5
        pre=np.zeros(y.size)
        cnt=0
        for i in range(y.size):
            if (self.sigmoid(np.dot(X[i],theta))>=thereshold) :
                pre[i]=1
            else :
                pre[i]=0
        for i in range(y.size):
            if (y[i]==pre[i]):
                cnt=cnt+1.0
        return cnt/y.size

    def re_pr(self,X,y,theta):
        thereshold = 0.5
        pre = np.zeros(y.size)
        cnt = 0
        for i in range(y.size):
            if (self.sigmoid(np.dot(X[i], theta)) >= thereshold):
                pre[i] = 1
            else:
                pre[i] = 0
        tn=0
        tp=0
        fn=0
        fp=0
        for i in range(y.size):
            if ((y[i]==1)&(pre[i]==0)):
                fn=fn+1
            elif ((y[i]==1)&(pre[i]==1)):
                tp=tp+1
            elif((y[i]==0)&(pre[i]==0)):
                tn=tn+1
            else:
                fp=fp+1
        pr=tp/(tp+fp)
        rc=tp/(tp+fn)
        return pr,rc