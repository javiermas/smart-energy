import logging
from pandas import DataFrame
from datetime import timedelta
from scipy.stats import norm
from numpy import nanmean, nansum, nan

from .base import Environment


class SBEnvironment(Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translation_dict = {
            'Battery': 'u8StateOfBattery',
            'Generator': 'fIPV_avg',
            'Consumer': 'fILoadDirect_avg',
        }

    def run(self, steps):
        self.initialize()
        for _ in range(steps):
            self.step()

    def initialize(self):
        logging.info('Initializing environment')
        self.mirror_repo.drop()
        start = self.source_repo.load_first_measurement()[0]['datetime']
        self._transfer_data(start, self.init_t)
        self.network.initialize()
        self.ml_service.initialize()
        for step in range(self.init_steps):
            self._transfer_data(self.init_t + (self.step_size * step),
                                self.init_t + (self.step_size * (step + 1)))
            self.network.update()
            readings = self.network.get_reading()
            data = self.readings_to_data(readings)
            actions = self.ml_service.get_action(data, random=True)
            self.network.interact(actions)
            reward = self.get_reward()
            self.ml_service.feed_reward(reward)
            if step % 50 == 0:
                logging.info(f'{step} memories created')

        self.mirror_repo.drop()
        self._transfer_data(start, self.init_t + (self.init_steps * self.step_size))
        self.network.initialize()
        self.ml_service.initialize()

    def _transfer_data(self, start, end):
        data = self.source_repo.load_data_within(start, end)
        self.mirror_repo.insert_many(data.to_dict('records'))

    def step(self):
        self.t += self.step_size
        self.network.update()
        readings = self.network.get_reading()
        data = self.readings_to_data(readings)
        actions = self.ml_service.get_action(data)
        self.network.interact(actions)
        reward = self.get_reward()
        self.ml_service.store_memory(actions, reward)

    def get_reward(self):
        readings = self.network.get_reading()
        gaussian_reward = self.get_gaussian_reward(readings)
        excess_battery_reward = self.get_excess_battery_reward(readings)
        empty_battery_reward = self.get_empty_battery_reward(readings)
        reward = gaussian_reward + excess_battery_reward + empty_battery_reward
        print('Battery state')
        print([installation['Battery'] for installation in readings.values()
               if installation['Battery'] is not None])
        print('Generation')
        print(nanmean([installation['Generator'] for installation in readings.values()
               if installation['Generator'] is not None]))
        print('Consumption')
        print(nanmean([installation['Consumer'] for installation in readings.values()
               if installation['Consumer'] is not None]))
        return reward
    
    def get_gaussian_reward(self, readings):
        return nanmean([nan if installation['Battery'] is None else norm(65, 15).pdf(installation['Battery'])
                        for installation in readings.values()]) * 100

    def get_excess_battery_reward(self, readings):
        return nansum([nan if installation['Battery'] is None else (installation['Battery'] > 95) * -1
                       for installation in readings.values()])

    def get_empty_battery_reward(self, readings):
        return nansum([nan if installation['Battery'] is None else (installation['Battery'] < 5) * -1
                       for installation in readings.values()])

    def readings_to_data(self, readings):
        readings_list = []
        for i, r in readings.items():
            reading_dict = {f'{self.translation_dict[elem]}': reading for elem, reading in r.items()}
            reading_dict['solbox_id'] = i.split('_')[1]
            reading_dict['datetime'] = self.t
            readings_list.append(reading_dict)

        return DataFrame(readings_list)
