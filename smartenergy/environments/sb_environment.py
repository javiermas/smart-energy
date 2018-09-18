import sys
import logging
from time import sleep
from pandas import DataFrame

from .base import Environment


class SBEnvironment(Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_up_logging()
        self.translation_dict = {
            'Battery': 'battery_state_discrete',
            'Generator': 'energy_generation_i',
            'Consumer': 'energy_consumption_i',
        }
        # TODO: rethink the different phases
        self.start = self.source_repo.load_first_measurement()[0]['datetime']
        self.burning_end = self.start + self.burning_steps * self.step_size
        self.init_end = self.burning_end + self.init_steps * self.step_size
        self.t = self.burning_end

    def run(self, steps):
        self.initialize()
        logging.info(f'Running environment for {steps} steps')
        sleep(5)
        for _ in range(steps):
            self.step()

    def initialize(self):
        logging.info('Initializing environment')
        self.mirror_repo.drop()
        self._transfer_data(self.start, self.burning_end)
        self.network.initialize()
        self.ml_service.initialize()
        for step in range(self.init_steps):
            print(f'---------- {self.t} ----------')
            self._transfer_data(self.t, self.t + self.step_size)
            self.network.update()
            readings = self.network.get_reading()
            data = self.readings_to_data(readings)
            actions = self.ml_service.get_action(data, random=True)
            self.network.interact(actions)
            reward = self.metrics_manager.get_reward(readings)
            self.ml_service.feed_reward(reward)
            if (step + 1) % 50 == 0:
                logging.info(f'{step+1} memories created')

            self.t += self.step_size

        self.mirror_repo.drop()
        self._transfer_data(self.start, self.init_end)
        self.network.initialize()
        self.ml_service.initialize()
        logging.info('Initialization finished')
        sleep(5)

    def _transfer_data(self, start, end):
        data = self.source_repo.load_data_within(start, end)
        self.mirror_repo.insert_many(data.to_dict('records'))

    def step(self):
        print(f'---------- {self.t} ----------')
        self._transfer_data(self.t, self.t + self.step_size)
        self.network.update()
        readings = self.network.get_reading()
        data = self.readings_to_data(readings)
        actions = self.ml_service.get_action(data)
        self.network.interact(actions)
        reward = self.metrics_manager.get_reward(readings)
        self.ml_service.feed_reward(reward)
        self.metrics_manager.update_metrics(readings)
        self.t += self.step_size
        if self.t.weekday() == 0 and self.t.hour == 0:
            self.get_weekly_report()

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

    def set_up_logging(self):
        logging.basicConfig(stream=sys.stdout,
                            level='DEBUG',
                            format='self.t:%(levelname)s:%(asctime)s:%(name)s:::%(message)s')
