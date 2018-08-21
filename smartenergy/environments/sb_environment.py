import logging
from pandas import DataFrame
from .base import Environment


class SBEnvironment(Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.translation_dict = {
            'Battery': 'u8StateOfBattery',
            'Generator': 'fILoadDirect_avg',
            'Consumer': 'fIPV_avg',
        }

    def run(self, steps):
        self.initialize()
        for _ in range(steps):
            self.step()

    def initialize(self):
        self.mirror_repo.drop()
        self._transfer_data()
        self.network.initialize()
        self.ml_service.train()

    def _transfer_data(self):
        data = self.source_repo.load_until(self.init_t)
        self.mirror_repo.insert_many(data.to_dict('records'))
        logging.info(f'Successfully transferred {data.shape[0]} data points')

    def step(self):
        self.t += self.step_size
        self.network.update()
        readings = self.network.get_reading()
        data = self.readings_to_data(readings)
        actions = self.ml_service.get_action(data)
        self.network.interact(actions)

    def get_excess_battery(self, readings):
        return 1

    def readings_to_data(self, readings):
        readings_list = []
        for i, r in readings.items():
            reading_dict = {f'{self.translation_dict[elem]}': reading for elem, reading in r.items()}
            reading_dict['solbox_id'] = i.split('_')[1]
            reading_dict['datetime'] = self.t
            readings_list.append(reading_dict)

        return DataFrame(readings_list)
