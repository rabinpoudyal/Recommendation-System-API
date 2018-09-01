"""
Request Handlers
"""

import tornado.web
from tornado import concurrent
from tornado import gen
from concurrent.futures import ThreadPoolExecutor

from app.base_handler import BaseApiHandler
from app.settings import MAX_MODEL_THREAD_POOL
from app.settings import MODEL_DIR
import pandas as pd
import numpy as np


class IndexHandler(tornado.web.RequestHandler):
    """APP is live"""

    def get(self):
        self.write("App is Live!")

    def head(self):
        self.finish()

class RecommendationPredictionHandler(BaseApiHandler):
    _thread_pool = ThreadPoolExecutor(max_workers=MAX_MODEL_THREAD_POOL)

    def initialize(self, *args, **kwargs):
        #self.user_factor = user_factor
        #self.item_factor = item_factor
        super().initialize(*args, **kwargs)

    @concurrent.run_on_executor(executor='_thread_pool')
    def _blocking_predict(self, X):
        user_factor = pd.read_pickle("{}/user_factor.pkl".format(MODEL_DIR))
        item_factor = pd.read_pickle("{}/item_factor.pkl".format(MODEL_DIR))
        #print(X[0][1])
        predictItemRating=pd.DataFrame(np.dot(user_factor.loc[X[0][0]],item_factor.T),index=item_factor.index,columns=['Rating'])
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:X[0][1]]
        # We found the ratings of all movies by the active user and then sorted them to find the top 3 movies 
        #topRecommendationTitles=moviesData.loc[moviesData.itemid.isin(topRecommendations.index)]
        #print(list(topRecommendationTitles.title))
        #print(topRecommendationTitles)
        return(topRecommendations.index.tolist())


    @gen.coroutine
    def predict(self, data):
        if type(data) == dict:
            data = [data]

        X = []
        for item in data:
            record  = (item.get("activeUser"), item.get("numberOfRec"))
            X.append(record)

        results = yield self._blocking_predict(X)
        self.respond(results)



class IrisPredictionHandler(BaseApiHandler):

    _thread_pool = ThreadPoolExecutor(max_workers=MAX_MODEL_THREAD_POOL)

    def initialize(self, model, *args, **kwargs):
        self.model = model
        super().initialize(*args, **kwargs)

    @concurrent.run_on_executor(executor='_thread_pool')
    def _blocking_predict(self, X):
        target_values = self.model.predict(X)
        target_names = ['setosa', 'versicolor', 'virginica']
        results = [target_names[pred] for pred in target_values]
        return results


    @gen.coroutine
    def predict(self, data):
        if type(data) == dict:
            data = [data]

        X = []
        for item in data:
            record  = (item.get("sepal_length"), item.get("sepal_width"), \
                    item.get("petal_length"), item.get("petal_width"))
            X.append(record)

        results = yield self._blocking_predict(X)
        self.respond(results)
