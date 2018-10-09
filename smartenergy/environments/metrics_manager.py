from numpy import nanmean, nansum, nan
from scipy.stats import norm


class MetricsManager(object):

    def __init__(self, source_repo):
        self.source_repo = source_repo
        self.excess_real = 0
        self.excess_simulation = 0

    def update_metrics(self, readings):
        self.excess_real += nanmean([self.source_repo.get_last_excess_energy_measurement(i)
                                     for i in self.source_repo.station_ids])
        self.excess_simulation += nanmean([installation['Generator'] for installation in readings.values()
                                           if installation['Battery'] == 100])

    def get_cumulative_reward(self, readings, next_state_values, next_state_weight):
        print('Battery state')
        print([installation['Battery'] for installation in readings.values()
               if installation['Battery'] is not None])
        print('Generation')
        print([round(installation['Generator'], 2) for installation in readings.values()
               if installation['Generator'] is not None])
        print('Consumption')
        print([round(installation['Consumer'], 2) for installation in readings.values()
               if installation['Consumer'] is not None])
        cumulative_reward = {
            name: (0 if reading['Battery'] is None else self._get_cumulative_reward_lambda(
                reading['Battery'], next_state_values[name], next_state_weight))
            for name, reading in readings.items()
        }
        return cumulative_reward
    
    def _get_cumulative_reward_lambda(self, state, next_state_value, next_state_weight):
        current_state_reward = sum([
            self.get_gaussian_reward(state),
            self.get_empty_battery_reward(state),
            self.get_excess_battery_reward(state)
        ])
        next_state_reward = next_state_weight * next_state_value
        return current_state_reward + next_state_reward

    @staticmethod
    def get_gaussian_reward(reading):
        return norm(65, 15).pdf(reading) * 100

    @staticmethod
    def get_excess_battery_reward(reading):
        return (reading > 95) * -1

    @staticmethod
    def get_empty_battery_reward(reading):
        return (reading < 5) * -1
