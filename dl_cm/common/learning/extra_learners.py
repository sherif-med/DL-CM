from abc import ABC, abstractmethod

from dl_cm.common.learning import LearnersFactory
from dl_cm.common.learning.base_learner import BaseLearner
from dl_cm.common.typing import StepInputStruct, StepOutputStruct, namedEntitySchema


class MultiLearner(BaseLearner, ABC):
    def __init__(
        self, learners: list[BaseLearner | namedEntitySchema], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.learners: list[BaseLearner] = LearnersFactory.create(learners)

    @abstractmethod
    def aggregate_output(self, outputs: list[StepOutputStruct]) -> StepOutputStruct:
        pass

    def forward(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        leraners_output: list[StepOutputStruct] = []
        for learner in self.learners:
            leraners_output.append(learner.forward(batch, *args, **kwargs))
        return self.aggregate_output(leraners_output)


class SequentialLearner(BaseLearner):
    def __init__(
        self, learners: list[BaseLearner | namedEntitySchema], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.learners: list[BaseLearner] = LearnersFactory.create(learners)

    def forward(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        for learner in self.learners:
            batch = learner.forward(batch, *args, **kwargs)
        return batch


class learnerWrapper(BaseLearner, ABC):
    def __init__(
        self, wraped_learner: BaseLearner | namedEntitySchema, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.wraped_learner: BaseLearner = LearnersFactory.create(wraped_learner)

    @abstractmethod
    def pre_step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        pass

    @abstractmethod
    def post_step(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        pass

    def forward(self, batch: StepInputStruct, *args, **kwargs) -> StepOutputStruct:
        batch = self.pre_step(batch, *args, **kwargs)
        batch = self.wraped_learner.forward(batch, *args, **kwargs)
        batch = self.post_step(batch, *args, **kwargs)
        return batch
