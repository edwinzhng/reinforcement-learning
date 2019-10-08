class ReplayExperience:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.actions = []
        self.observations = []
        self.rewards = []
        self.terminals = []

    def add(self, action, observation, reward, terminal):
        pass
