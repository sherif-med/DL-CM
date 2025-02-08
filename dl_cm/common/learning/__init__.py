from dl_cm.utils.registery import Registry
from dl_cm.utils.ppattern.factory import BaseFactory
from dl_cm.common.learning.base_learner import BaseLearner

LEARNERS_REGISTERY = Registry("Learners")

class LearnersFactory(BaseFactory):

    @staticmethod
    def base_class()-> type:
        return BaseLearner
    