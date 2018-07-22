from abc import abstractmethod


class MLService(object):

    def __init__(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def get_action(self, readings):
        return {i_name: {e: 1 for e in i.keys()} for i_name, i in self.action_space.items()}

