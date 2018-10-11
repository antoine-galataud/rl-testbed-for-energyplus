import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import math

"""
Utility methods to print training episode statistics and plot them
"""
class EpisodeStatsUtils:

    def show_statistics(self, title, series):
        print('{:25} ave={:5,.2f}, min={:5,.2f}, max={:5,.2f}, std={:5,.2f}'.format(title, np.average(series), np.min(series), np.max(series), np.std(series)))

    def get_statistics(self, series):
        return np.average(series), np.min(series), np.max(series), np.std(series)

    def show_distrib(self, title, series):
        import pandas as pd
        
        df = pd.DataFrame(series)
        print(df.describe())

        dist = [0 for i in range(1000)]
        for v in series:
            idx = int(math.floor(v * 10))
            if idx >= 1000:
                idx = 999
            if idx < 0:
                idx = 0
            dist[idx] += 1
        print(title)
        print('    degree 0.0-0.9 0.0   0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9')
        print('    -------------------------------------------------------------------------')
        for t in range(170, 280, 10):
            print('    {:4.1f}C {:5.1%}  '.format(t / 10.0, sum(dist[t:(t+10)]) / len(series)), end='')
            for tt in range(t, t + 10):
                print(' {:5.1%}'.format(dist[tt] / len(series)), end='')
            print('')
        
    def plot_episode(self, model, ep, *actions):
        print('===========================================')
        print('EPISODE {}'.format(ep))
        model.read_episode(ep)

        self.show_statistics('Reward', model.rewards)
        self.show_statistics('westzone_temp', model.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'])
        self.show_statistics('eastzone_temp', model.df['EAST ZONE:Zone Air Temperature [C](TimeStep)'])
        self.show_statistics('Power consumption', model.df['Whole Building:Facility Total Electric Demand Power [W](Hourly)'])
        self.show_statistics('pue', model.pue)
        self.show_distrib('westzone_temp distribution', model.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'])
        self.show_distrib('eastzone_temp distribution', model.df['EAST ZONE:Zone Air Temperature [C](TimeStep)'])

        model.axepisode = []
        for i in range(model.num_axes):
            ax = model.fig.add_axes([0.05, 1.00 - 0.70 / model.num_axes * (i + 1), 0.90, 0.12])
            ax.set_xmargin(0)
            model.axepisode.append(ax)
            ax.set_xticks(model.x_pos)
            ax.set_xticklabels(model.x_labels)
            ax.tick_params(labelbottom='off')
            ax.grid(True)

        idx = 0
        show_west = True

        if True:
            # Plot zone and outdoor temperature
            ax = model.axepisode[idx]
            idx += 1
            ax.lines = []
            ax.plot(model.westzone_temp, 'C0', label='Westzone temperature')
            ax.plot(model.eastzone_temp, 'C1', label='Eastzone temperature')
            ax.plot(model.outdoor_temp, 'C2', label='Outdoor temperature')
            ax.plot(model.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(model.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(-1.0, 40.0)

        if True:
            # Plot return air and sestpoint temperature
            ax = model.axepisode[idx]
            idx += 1
            ax.lines = []
            if show_west:
                ax.plot(model.westzone_return_air_temp, 'C0', label='WEST ZONE RETURN AIR NODE:System Node Temperature')
                ax.plot(model.westzone_dec_outlet_setpoint_temp, 'C1', label='Westzone DEC outlet setpoint temperature')
            else:
                #ax.plot(model.eastzone_return_air_temp, 'C0', label='EAST ZONE RETURN AIR NODE:System Node Temperature')
                ax.plot(model.eastzone_dec_outlet_setpoint_temp, 'C1', label='Eastzone DEC outlet setpoint temperature')
            ax.plot(model.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(model.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(0.0, 30.0)

        if True:
            # Plot west zone, return air, mixed air, supply fan
            ax = model.axepisode[idx]
            idx += 1
            ax.lines = []
            if show_west:
                ax.plot(model.westzone_return_air_temp, 'C0', label='WEST ZONE RETURN AIR NODE:System Node Temperature')
                ax.plot(model.westzone_mixed_air_temp, 'C1', label='WEST ZONE MIXED AIR NODE:System Node Temperature')
                ax.plot(model.westzone_supply_fan_outlet_temp, 'C2', label='WEST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature')
                ax.plot(model.westzone_dec_outlet_temp, 'C3', label='Westzone DEC outlet temperature')
            else:
                #ax.plot(model.eastzone_return_air_temp, 'C0', label='EAST ZONE RETURN AIR NODE:System Node Temperature')
                #ax.plot(model.eastzone_mixed_air_temp, 'C1', label='EAST ZONE MIXED AIR NODE:System Node Temperature')
                ax.plot(model.eastzone_supply_fan_outlet_temp, 'C2', label='EAST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature')
                ax.plot(model.eastzone_dec_outlet_temp, 'C3', label='Eastzone DEC outlet temperature')
            ax.plot(model.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(model.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(0.0, 30.0)

        if True:
            # Plot west zone ccoil, air loop
            ax = model.axepisode[idx]
            idx += 1
            ax.lines = []
            if show_west:
                ax.plot(model.westzone_iec_outlet_temp, 'C0', label='Westzone IEC outlet temperature')
                ax.plot(model.westzone_ccoil_air_outlet_temp, 'C1', label='Westzone ccoil air outlet temperature')
                ax.plot(model.westzone_air_loop_outlet_temp, 'C2', label='Westzone air loop outlet temperature')
                ax.plot(model.westzone_dec_outlet_setpoint_temp, label='Westzone DEC outlet setpoint temperature', linewidth=0.5, color='gray')
            else:
                ax.plot(model.eastzone_iec_outlet_temp, 'C0', label='Eastzone IEC outlet temperature')
                ax.plot(model.eastzone_ccoil_air_outlet_temp, 'C1', label='Eastzone ccoil air outlet temperature')
                ax.plot(model.eastzone_air_loop_outlet_temp, 'C2', label='Eastzone air loop outlet temperature')
                ax.plot(model.eastzone_dec_outlet_setpoint_temp, label='Eastzone DEC outlet setpoint temperature', linewidth=0.5, color='gray')
            ax.plot(model.cooling_setpoint, label='Cooling setpoint', linestyle='--', linewidth=0.5, color='blue')
            ax.plot(model.heating_setpoint, label='Heating setpoint', linestyle='--', linewidth=0.5, color='red')
            ax.legend()
            ax.set_ylabel('Temperature (C)')
            ax.set_ylim(0.0, 30.0)

        if False:
            # Plot calculated reward
            ax = model.axepisode[idx]
            idx += 1
            ax.lines = []
            ax.plot(model.rewards, 'C0', label='Reward')
            ax.plot(model.rewards_gaussian1, 'C1', label='Gaussian1')
            ax.plot(model.rewards_trapezoid1, 'C2', label='Trapezoid1')
            ax.plot(model.rewards_gaussian2, 'C3', label='Gaussian2')
            ax.plot(model.rewards_trapezoid2, 'C4', label='Trapezoid2')
            ax.plot(model.rewards_power, 'C5', label='Power')
            ax.legend()
            ax.set_ylabel('Reward')
            ax.set_ylim(-2.0, 2.0)

        if False:
            # Plot PUE
            ax = model.axepisode[idx]
            idx += 1
            ax.lines = []
            ax.plot(model.pue, 'C0', label='PUE')
            ax.legend()
            ax.set_ylabel('PUE')
            ax.set_ylim(top=2.0, bottom=1.0)
            
        if False:
            # Plot other electric power consumptions
            ax = model.axepisode[idx]
            idx += 1
            ax.lines = []
            #ax.plot(model.total_electric_demand_power, 'C0', label='Whole Building:Facility Total Electric Demand Power')
            #ax.plot(model.total_building_electric_demand_power, 'C1', label='Whole Building:Facility Total Building Electric Demand Power')
            #ax.plot(model.total_hvac_electric_demand_power, 'C2', label='Whole Building:Facility Total HVAC Electric Demand Power')
            for i, pow in enumerate(model.electric_powers):
                ax.plot(model.df[pow], 'C{}'.format(i % 10), label=pow)
            ax.legend()
            ax.set_ylabel('Power (W)')
            ax.set_xlabel('Simulation steps')
            ax.tick_params(labelbottom='on')

        if True:
            # Plot power consumptions
            ax = model.axepisode[idx]
            idx += 1
            ax.lines = []
            ax.plot(model.total_electric_demand_power, 'C0', label='Whole Building:Facility Total Electric Demand Power')
            ax.plot(model.total_building_electric_demand_power, 'C1', label='Whole Building:Facility Total Building Electric Demand Power')
            ax.plot(model.total_hvac_electric_demand_power, 'C2', label='Whole Building:Facility Total HVAC Electric Demand Power')
            ax.legend()
            ax.set_ylabel('Power (W)')
            ax.set_xlabel('Simulation days (MM/DD)')
            ax.tick_params(labelbottom='on')

        if 'fan' in actions and True:
            # Plot return air and sestpoint temperature
            ax = model.axepisode[idx]
            idx += 1
            ax.lines = []
            ax.plot(model.df['WEST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)'], 'C0', label='WEST ZONE Air Mass Flow Rate')
            ax.plot(model.df['EAST ZONE SUPPLY FAN:Fan Air Mass Flow Rate [kg/s](Hourly)'], 'C1', label='EAST ZONE Air Mass Flow Rate')
            ax.legend()
            ax.set_ylabel('Air volume rate (kg/s)')
            ax.set_ylim(0.0, 10.0)

        # Show average power consumption in text
        if model.text_power_consumption is not None:
            model.text_power_consumption.remove()
        model.text_power_consumption = model.fig.text(0.02,  0.25, 'Whole Power:    {:6,.1f} kW'.format(np.average(model.df['Whole Building:Facility Total Electric Demand Power [W](Hourly)']) / 1000))
        model.text_power_consumption = model.fig.text(0.02,  0.235, 'Building Power: {:6,.1f} kW'.format(np.average(model.df['Whole Building:Facility Total Building Electric Demand Power [W](Hourly)']) / 1000))
        model.text_power_consumption = model.fig.text(0.02,  0.22, 'HVAC Power:     {:6,.1f} kW'.format(np.average(model.df['Whole Building:Facility Total HVAC Electric Demand Power [W](Hourly)']) / 1000))
