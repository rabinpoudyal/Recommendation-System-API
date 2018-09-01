import pandas as pd
import numpy as np
import pickle
    
def matrixFactorization(R, K, steps=10, gamma=0.001,lamda=0.02):
    # R is the user item rating matrix 
    # K is the number of factors we will find 
    # We'll be using Stochastic Gradient descent to find the factor vectors 
    # steps, gamma and lamda are parameters the SGD will use - we'll get to them
    # in a bit 
    N=len(R.index)# Number of users
    M=len(R.columns) # Number of items 
    P=pd.DataFrame(np.random.rand(N,K),index=R.index)
    # This is the user factor matrix we want to find. It will have N rows 
    # on for each user and K columns, one for each factor. We are initializing 
    # this matrix with some random numbers, then we will iteratively move towards 
    # the actual value we want to find 
    Q=pd.DataFrame(np.random.rand(M,K),index=R.columns)
    # This is the product factor matrix we want to find. It will have M rows, 
    # one for each product/item/movie. 
    for step in range(steps):
        # SGD will loop through the ratings in the user item rating matrix 
        # It will do this as many times as we specify (number of steps) or 
        # until the error we are minimizing reaches a certain threshold 
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    # For each rating that exists in the training set 
                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])
                    # This is the error for one rating 
                    # ie difference between the actual value of the rating 
                    # and the predicted value (dot product of the corresponding 
                    # user factor vector and item-factor vector)
                    # We have an error function to minimize. 
                    # The Ps and Qs should be moved in the downward direction 
                    # of the slope of the error at the current point 
                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    # Gamma is the size of the step we are taking / moving the value
                    # of P by 
                    # The value in the brackets is the partial derivative of the 
                    # error function ie the slope. Lamda is the value of the 
                    # regularization parameter which penalizes the model for the 
                    # number of factors we are finding. 
                    Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
        # At the end of this we have looped through all the ratings once. 
        # Let's check the value of the error function to see if we have reached 
        # the threshold at which we want to stop, else we will repeat the process
        e=0
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    #Sum of squares of the errors in the rating
                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
                    #print(e)
        if e<0.001:
            break
        print(step, e)
    return P,Q


def run():
    dataFile = 'data/users.movies.csv'
    dataset = pd.read_csv(dataFile, sep="|", header=None, names=['userid', 'itemid','rating'], usecols=[0,1,2])
    moviesFile = 'data/u.item.csv'
    moviesData = pd.read_csv(moviesFile, sep="|", header=None, index_col=False, names=['itemid','title'], usecols=[0,1], encoding='latin' )
    
    data = pd.merge(moviesData,dataset, left_on='itemid', right_on='itemid')
    
    userItemRatingMatrix = pd.pivot_table(data, values='rating', index=['userid'], columns=['itemid'])
    (P,Q)=matrixFactorization(userItemRatingMatrix.iloc[:100,:100],K=2,gamma=0.001,lamda=0.02, steps=1)
    activeUser=1
    
    predictItemRating=pd.DataFrame(np.dot(P.loc[activeUser],Q.T),index=Q.index,columns=['Rating'])
    topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:3]
    # We found the ratings of all movies by the active user and then sorted them to find the top 3 movies 
    topRecommendationTitles=moviesData.loc[moviesData.itemid.isin(topRecommendations.index)]
    print(list(topRecommendationTitles.title))
    P.to_pickle('../models/user_factor.pkl')
    Q.to_pickle('../models/item_factor.pkl')

    
    

if __name__ =='__main__':
    run()