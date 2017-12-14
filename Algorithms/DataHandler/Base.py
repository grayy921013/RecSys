"""
     Content Based Algorithms Base Class
===================================================

All CBAlgorithms should inherit from this class and included the methods here defined

"""

# Author: Caleb De La Cruz P. <delacruzp>


import logging
from abc import ABCMeta, abstractmethod

from Util import Field

logger = logging.getLogger(__name__)


class DataHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_data(self, field):
        '''
        Look in the specified dataset for the given field and return a list with the value for this field for every
        item in the dataset
        :param field: A value of the enum Field
        :return: array of strings
        '''
        if not isinstance(field, Field):
            raise AttributeError("Parameter field should be value of the enum Field")
