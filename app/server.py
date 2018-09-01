# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import logging.config

import tornado.ioloop
import tornado.web
from tornado.options import options

from sklearn.externals import joblib

from app.settings import MODEL_DIR
from app.handler import IndexHandler, RecommendationPredictionHandler
import pandas as pd


MODELS = {}


def load_model(pickle_filename):
    return joblib.load(pickle_filename)


def main():

    # Get the Port and Debug mode from command line options or default in settings.py
    options.parse_command_line()


    # create logger for app
    logger = logging.getLogger('app')
    logger.setLevel(logging.INFO)

    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT)

    # Load ML Models
    logger.info("Loading Recommendation System Prediction Model...")
    #MODELS["user_factor"] = pd.read_pickle(MODEL_DIR, "user_factor.pkl")
    #MODELS["item_factor"] = pd.read_pickle(MODEL_DIR, "item_factor.pkl")

    urls = [
        (r"/$", IndexHandler),
        (r"/api/recommendation/(?P<action>[a-zA-Z]+)?", RecommendationPredictionHandler)
            #dict(user_factor=MODELS["user_factor"], item_factor=MODELS["item_factor"]))
    ]

    # Create Tornado application
    application = tornado.web.Application(
        urls,
        debug=options.debug,
        autoreload=options.debug)

    # Start Server
    logger.info("Starting App on Port: {} with Debug Mode: {}".format(options.port, options.debug))
    application.listen(options.port)
    tornado.ioloop.IOLoop.current().start()


