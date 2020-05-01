import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from tqdm import tqdm
from logger import logger

from optimizer import Optimizer


class gb_rf_evolution:
    def __init__(self, generations, population, params):
        """
        :param generations int: number of generation
        :param population int: number of population inside each generation
        :param params dict: dictionary with parameters
        """
        self._generations = generations
        self._population = population
        self._params = params
        self.networks = None
        self.best_params = None

    def evolve(self, x_train, y_train, x_test, y_test):
        """
        Takes data for traning and data for test and iterate thought generations to find parameters with lowest error
        :param x_train array: array with features for traning
        :param y_train array: array with real values for traning
        :param x_test array: array with features for test
        :param y_test array: array with real values for test
        :return: None
        """
        optimizer = Optimizer(self._params)
        self._networks = list(optimizer.create_population(self._population))

        models = []
        params = []
        scores = []
        for i in range(self._generations - 1):
            generation_scores, trained = self._train_networks(x_train, y_train, x_test, y_test)
            self._networks = optimizer.evolve(self._networks)
            models.extend(trained)
            params.extend(self._networks)
            scores.extend(generation_scores)


        params = sorted(params, key=lambda x: x.accuracy, reverse=True)
        self.best_params = params[0]
        logger.info("best accuracy: {}, best params: {}".format(self.best_params.accuracy, self.best_params.network))
        return scores, models, params
    
    def _train_networks(self, x_train, y_train, x_test, y_test):
        """
        Method for networks training
        :param x_train array: array with features for traning
        :param y_train array: array with real values for traning
        :param x_test array: array with features for test
        :param y_test array: array with real values for test
        :return: None
        """
        
        models = []
        scores = []
        pbar = tqdm(total=len(self._networks))
        for network in self._networks:
            r2, model = network.train(x_train, y_train, x_test, y_test)
            pbar.update(1)
            models.append(model)
            scores.append(r2)
        pbar.close()
        return scores, models

    def _get_average_accuracy(self, networks):
        """
        :param networks list: list of dictionaries
        :return float: mean accuracy per population
        """
        return sum([network.accuracy for network in networks]) / len(networks)
