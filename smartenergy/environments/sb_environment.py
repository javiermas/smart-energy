from time import sleep
from pandas import DataFrame

from .base import Environment
from ..database import Performance

DEFAULT_HYPERPARAMETERS = {
    'next_state_weight': 0.1
}


class SBEnvironment(Environment):

    def __init__(self, hyperparameters=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translation_dict = {
            'Battery': 'battery_state_discrete',
            'Generator': 'energy_generation_computed_i',
            'Consumer': 'energy_consumption_computed_i',
        }
        self.start = self.source_repo.load_first_measurement()[0]['datetime']
        self.end = self.source_repo.load_last_measurement()[0]['datetime']
        self.burning_end = self.start + self.burning_steps * self.step_size
        self.init_end = self.burning_end + self.init_steps * self.step_size
        self.t = self.burning_end
        self.hyperparameters = hyperparameters or DEFAULT_HYPERPARAMETERS
        self.next_state_weight = self.hyperparameters['next_state_weight']
        self.next_state_weight = 0.1
        self.training_frequency_steps = 10
        self.performance_repo = Performance()
        self.round = 0

    def run(self, steps):
        self.initialize()
        self.log.info(f'Running environment for {steps} steps')
        sleep(5)
        for i in range(steps):
            if (i + 1) % self.training_frequency_steps == 0:
                loss = self.ml_service.train()
                self.performance_repo.insert_one({**{'t': self.t}, **loss})

            self.step(random=False)
            if self.t.weekday() == 0 and self.t.hour == 0:
                self.get_weekly_report()

            self.t += self.step_size
            if self.t == self.end:
                self.log.info(f'Round {self.round} completed')
                self.restart()
                self.round += 1

    def restart(self):
        self.t = self.init_end
        self.mirror_repo.drop()
        self._transfer_data(self.start, self.init_end)
        self.network.initialize()

    def initialize(self):
        self.log.info('Initializing environment')
        self.mirror_repo.drop()
        self.performance_repo.drop()
        self._transfer_data(self.start, self.burning_end)
        self.network.initialize()
        self.ml_service.initialize()
        self.t = self.burning_end
        for step in range(self.init_steps):
            self.step(random=True)
            if (step + 1) % 50 == 0:
                self.log.info(f'{step+1} memories created')

            self.t += self.step_size

        self.mirror_repo.drop()
        self._transfer_data(self.start, self.init_end)
        self.network.initialize()
        self.log.info('Initialization finished')
        sleep(5)

    def _transfer_data(self, start, end):
        data = self.source_repo.load_data_within(start, end)
        self.mirror_repo.insert_many(data.to_dict('records'))

    def step(self, random):
        print(f'---------- {self.t} ----------')
        self._transfer_data(self.t, self.t + self.step_size)
        self.network.update()
        readings = self.network.get_reading()
        data = self.readings_to_data(readings)
        actions = self.ml_service.get_action(data, random)
        self.network.interact(actions)
        readings_next = self.readings_to_data(self.network.get_reading())
        next_state_value = self.ml_service.get_state_value(readings_next)
        reward = self.metrics_manager.get_cumulative_reward(readings, next_state_value,
                                                            self.next_state_weight)
        self.ml_service.feed_reward(reward)

    def readings_to_data(self, readings):
        readings_list = []
        for i, r in readings.items():
            reading_dict = {f'{self.translation_dict[elem]}': reading for elem, reading in r.items()}
            reading_dict['solbox_id'] = i.split('_')[1]
            reading_dict['datetime'] = self.t
            readings_list.append(reading_dict)

        return DataFrame(readings_list)

    def get_weekly_report(self):
        print('##############################################################################')
        print(f'{self.t} Weekly report')
        print(f'Real excess battery {self.metrics_manager.excess_real}')
        print(f'Simulation excess battery {self.metrics_manager.excess_simulation}')
        print('##############################################################################')
        sleep(10)
