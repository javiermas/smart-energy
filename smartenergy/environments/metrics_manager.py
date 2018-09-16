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

    def get_reward(self, readings):
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
