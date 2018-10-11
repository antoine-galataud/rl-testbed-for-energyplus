# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

import os
import time
import numpy as np
from scipy.special import expit
import pandas as pd
import datetime as dt
from gym import spaces
from gym_energyplus.envs.energyplus_model import EnergyPlusModel
from gym_energyplus.envs.energyplus_episode_stats import EpisodeStatsUtils
from glob import glob
from tqdm import tqdm

class EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp(EnergyPlusModel):
    
    def __init__(self,
                 model_file,
                 log_dir,
                 verbose=False):
        super(EnergyPlusModel2ZoneDataCenterHVAC_wEconomizer_Temp, self).__init__(model_file, log_dir, verbose)
        self.reward_low_limit = -10000.
        self.axepisode = None
        self.num_axes = 5
        self.text_power_consumption = None

        self.electric_powers = [
            #'Whole Building:Facility Total Electric Demand Power [W](Hourly)', # very high
            #'Whole Building:Facility Total Building Electric Demand Power [W](Hourly)', # very high
            #'Whole Building:Facility Total HVAC Electric Demand Power [W](Hourly)', # low

            #'WESTDATACENTER_EQUIP:ITE CPU Electric Power [W](Hourly)', # low
            #'WESTDATACENTER_EQUIP:ITE Fan Electric Power [W](Hourly)', # low
            #'WESTDATACENTER_EQUIP:ITE UPS Electric Power [W](Hourly)', # low

            #'WESTDATACENTER_EQUIP:ITE CPU Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'WESTDATACENTER_EQUIP:ITE Fan Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'WEST ZONE:Zone ITE CPU Electric Power [W](Hourly)', # low
            #'WEST ZONE:Zone ITE Fan Electric Power [W](Hourly)', # low
            #'WEST ZONE:Zone ITE UPS Electric Power [W](Hourly)', # low
            #'WEST ZONE:Zone ITE CPU Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'WEST ZONE:Zone ITE Fan Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'WEST DATA CENTER IEC:Evaporative Cooler Electric Power [W](TimeStep)', # low (works only on very cold day)
            #'WEST DATA CENTER DEC:Evaporative Cooler Electric Power [W](TimeStep)', # low (never works)
            #'EMS:Power Utilization Effectiveness [](TimeStep)',


            #'Whole Building:Facility Total Electric Demand Power [W](Hourly)', # very high
            #'Whole Building:Facility Total Building Electric Demand Power [W](Hourly)', # very high
            #'Whole Building:Facility Total HVAC Electric Demand Power [W](Hourly)', # low

            #'EASTDATACENTER_EQUIP:ITE CPU Electric Power [W](Hourly)', # low
            #'EASTDATACENTER_EQUIP:ITE Fan Electric Power [W](Hourly)', # low
            #'EASTDATACENTER_EQUIP:ITE UPS Electric Power [W](Hourly)', # low

            #'EASTDATACENTER_EQUIP:ITE CPU Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'EASTDATACENTER_EQUIP:ITE Fan Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'EAST ZONE:Zone ITE CPU Electric Power [W](Hourly)', # low
            #'EAST ZONE:Zone ITE Fan Electric Power [W](Hourly)', # low
            #'EAST ZONE:Zone ITE UPS Electric Power [W](Hourly)', # low
            #'EAST ZONE:Zone ITE CPU Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'EAST ZONE:Zone ITE Fan Electric Power at Design Inlet Conditions [W](Hourly)', # low (only depends on time in day)
            #'EAST DATA CENTER IEC:Evaporative Cooler Electric Power [W](TimeStep)', # low (works only on very cold day)
            #'EAST DATA CENTER DEC:Evaporative Cooler Electric Power [W](TimeStep)', # low (never works)
            
            #'EMS:Power Utilization Effectiveness [](TimeStep)',
        ]
        
    def setup_spaces(self):
        # Bound action temperature
        lo = 10.0
        hi = 40.0
        self.action_space = spaces.Box(low =   np.array([ lo, lo]),
                                       high =  np.array([ hi, hi]),
                                       dtype = np.float32)
        self.observation_space = spaces.Box(low =   np.array([-20.0, -20.0, -20.0,          0.0,          0.0,          0.0]),
                                            high =  np.array([ 50.0,  50.0,  50.0, 1000000000.0, 1000000000.0, 1000000000.0]),
                                            dtype = np.float32)
        
    def set_raw_state(self, raw_state):
        if raw_state is not None:
            self.raw_state = raw_state
        else:
            self.raw_state = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    
    def compute_reward(self):
        rew, _ = self._compute_reward()
        return rew

    def _compute_reward(self, raw_state = None):
        return self.compute_reward_center23_5_gaussian1_0_trapezoid0_1_pue0_0(raw_state)
        #return self.compute_reward_gaussian1_0_trapezoid1_0_pue0_0(raw_state)
        #return self.compute_reward_gaussian1_0_trapezoid0_1_pue0_0_pow0(raw_state)
        #return self.compute_reward_gaussian1_0_trapezoid1_0_pue0_0(raw_state)
        
    def compute_reward_center23_5_gaussian1_0_trapezoid0_1_pue0_0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 23.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 0.1,
            fluctuation_weight = 0.0,
            PUE_weight = 0.0,
            Whole_Building_Power_weight = 1 / 100000.0,
            raw_state = raw_state)
        
    def compute_reward_gaussian1_0_trapezoid1_0_pue0_0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 1.0,
            fluctuation_weight = 0.0,
            PUE_weight = 0.0,
            Whole_Building_Power_weight = 1 / 100000.0,
            raw_state = raw_state)

    def compute_reward_gaussian1_0_trapezoid0_1_pue0_0_pow0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 0.1,
            fluctuation_weight = 0.0,
            PUE_weight = 0.0,
            Whole_Building_Power_weight = 0.0,
            raw_state = raw_state)

    def compute_reward_gaussian1_0_trapezoid0_1_pue0_0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 0.1,
            fluctuation_weight = 0.0,
            PUE_weight = 0.0,
            Whole_Building_Power_weight = 1 / 100000.0,
            raw_state = raw_state)
    
    def compute_reward_gaussian_pue0_0(self, raw_state = None): # gaussian, PUE
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 1.0,
            temperature_trapezoid_weight = 0.,
            fluctuation_weight = 0.1,
            PUE_weight = 1.0,
            Whole_Building_Power_weight = 0.,
            raw_state = raw_state)
    
    def compute_reward_gaussian_whole_power(self, raw_state = None): # gaussian, whole power
        return self.compute_reward_common(
            temperature_center = 22.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 1.0,
            temperature_trapezoid_weight = 0.,
            fluctuation_weight = 0.1,
            PUE_weight = 0.0, # PUE not used
            Whole_Building_Power_weight = 1 / 100000.,
            raw_state = raw_state)
    
    def compute_reward_common(self, 
                              temperature_center = 22.5,
                              temperature_tolerance = 0.5,
                              temperature_gaussian_weight = 0.,
                              temperature_gaussian_sharpness = 1.,
                              temperature_trapezoid_weight = 0.,
                              fluctuation_weight = 0.,
                              PUE_weight = 0.,
                              Whole_Building_Power_weight = 0.,
                              raw_state = None):
        if raw_state is not None:
            st = raw_state
        else:
            st = self.raw_state

        Tenv = st[0]
        Tz1 = st[1]
        Tz2 = st[2]
        PUE = st[3]
        Whole_Building_Power = st[4]
        IT_Equip_Power = st[5]
        Whole_HVAC_Power = st[6]

        rew_PUE = -(PUE - 1.0) * PUE_weight
        # Temp. gaussian
        rew_temp_gaussian1 = np.exp(-(Tz1 - temperature_center) * (Tz1 - temperature_center) * temperature_gaussian_sharpness) * temperature_gaussian_weight
        rew_temp_gaussian2 = np.exp(-(Tz2 - temperature_center) * (Tz2 - temperature_center) * temperature_gaussian_sharpness) * temperature_gaussian_weight
        rew_temp_gaussian = rew_temp_gaussian1 + rew_temp_gaussian2
        # Temp. Trapezoid
        phi_low = temperature_center - temperature_tolerance
        phi_high = temperature_center + temperature_tolerance
        if Tz1 < phi_low:
            rew_temp_trapezoid1 = - temperature_trapezoid_weight * (phi_low - Tz1)
        elif Tz1 > phi_high:
            rew_temp_trapezoid1 = - temperature_trapezoid_weight * (Tz1 - phi_high)
        else:
            rew_temp_trapezoid1 = 0.
        if Tz2 < phi_low:
            rew_temp_trapezoid2 = - temperature_trapezoid_weight * (phi_low - Tz2)
        elif Tz2 > phi_high:
            rew_temp_trapezoid2 = - temperature_trapezoid_weight * (Tz2 - phi_high)
        else:
            rew_temp_trapezoid2 = 0.
        rew_temp_trapezoid = rew_temp_trapezoid1 + rew_temp_trapezoid2
        
        rew_fluct = 0.
        if raw_state is None:
            for cur, prev in zip(self.action, self.action_prev):
                rew_fluct -= abs(cur - prev) * fluctuation_weight
        rew_Whole_Building_Power = - Whole_Building_Power * Whole_Building_Power_weight
        rew = rew_temp_gaussian + rew_temp_trapezoid + rew_fluct + rew_PUE + rew_Whole_Building_Power
        if os.path.exists("/tmp/verbose"):
            print('compute_reward: rew={:7.3f} (temp_gaussian1={:7.3f}, temp_gaussian2={:7.3f}, temp_trapezoid1={:7.3f}, temp_trapezoid2={:7.3f}, fluct={:7.3f}, PUE={:7.3f}, Power={:7.3f})'.format(rew, rew_temp_gaussian1, rew_temp_gaussian2, rew_temp_trapezoid1, rew_temp_trapezoid2, rew_fluct, rew_PUE, rew_Whole_Building_Power))
        if os.path.exists("/tmp/verbose2"):
            print('compute_reward: Tenv={:7.3f}, Tz1={:7.3f}, Tz2={:7.3f}, PUE={:7.3f}, Whole_Powerd2={:8.1f}, ITE_Power={:8.1f}, HVAC_Power={:8.1f}, Act1={:7.3f}, Act2={:7.3f}'.format(Tenv, Tz1, Tz2, PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power, self.action[0], self.action[1]))
        
        return rew, (rew_temp_gaussian1, rew_temp_trapezoid1, rew_temp_gaussian2, rew_temp_trapezoid2, rew_Whole_Building_Power)
    
    # Performes mapping from raw_state (retrieved from EnergyPlus process as is) to gym compatible state
    #
    #   state[0] = raw_state[0]: Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)
    #   state[1] = raw_state[1]: WEST ZONE:Zone Air Temperature [C](TimeStep)
    #   state[2] = raw_state[2]: EAST ZONE:Zone Air Temperature [C](TimeStep)
    #              raw_state[3]: EMS:Power Utilization Effectiveness [](TimeStep)
    #   state[3] = raw_state[4]: Whole Building:Facility Total Electric Demand Power [W](Hourly)
    #   state[4] = raw_state[5]: Whole Building:Facility Total Building Electric Demand Power [W](Hourly)
    #   state[5] = raw_state[6]: Whole Building:Facility Total HVAC Electric Demand Power [W](Hourly)
    def format_state(self, raw_state):
        return np.array([raw_state[0], raw_state[1], raw_state[2], raw_state[4], raw_state[5], raw_state[6]])

    def read_episode(self, ep):
        if type(ep) is str:
            file_path = ep
        else:
            ep_dir = self.episode_dirs[ep]
            for file in ['eplusout.csv', 'eplusout.csv.gz']:
                file_path = ep_dir + '/' + file
                if os.path.exists(file_path):
                    break;
            else:
                print('No CSV or CSV.gz found under {}'.format(ep_dir))
                quit()
        print('read_episode: file={}'.format(file_path))
        df = pd.read_csv(file_path).fillna(method='ffill').fillna(method='bfill')
        self.df = df
        date = df['Date/Time']
        date_time = self._convert_datetime24(date)

        epw_files = glob(os.path.dirname(file_path) + '/USA_??_*.epw')
        if len(epw_files) == 1:
            self.weather_key = os.path.basename(epw_files[0])[4:6]
        else:
            self.weather_key = '  '
        self.outdoor_temp = df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
        self.westzone_temp = df['WEST ZONE:Zone Air Temperature [C](TimeStep)']
        self.eastzone_temp = df['EAST ZONE:Zone Air Temperature [C](TimeStep)']

        self.pue = df['EMS:Power Utilization Effectiveness [](TimeStep)']

        #self.westzone_ite_cpu_electric_power = df['WEST ZONE:Zone ITE CPU Electric Power [W](Hourly)']
        #self.westzone_ite_fan_electric_power = df['WEST ZONE:Zone ITE Fan Electric Power [W](Hourly)']
        #self.westzone_ite_ups_electric_power = df['WEST ZONE:Zone ITE UPS Electric Power [W](Hourly)']

        #WEST ZONE INLET NODE:System Node Temperature [C](TimeStep)
        #WEST ZONE INLET NODE:System Node Mass Flow Rate [kg/s](TimeStep)

        self.westzone_return_air_temp = df['WEST ZONE RETURN AIR NODE:System Node Temperature [C](TimeStep)']
        self.westzone_mixed_air_temp = df['WEST ZONE MIXED AIR NODE:System Node Temperature [C](TimeStep)']
        self.westzone_supply_fan_outlet_temp = df['WEST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_dec_outlet_temp = df['WEST ZONE DEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_dec_outlet_setpoint_temp = df['WEST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.westzone_iec_outlet_temp = df['WEST ZONE IEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_iec_outlet_setpoint_temp = df['WEST ZONE IEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.westzone_ccoil_air_outlet_temp = df['WEST ZONE CCOIL AIR OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_ccoil_air_outlet_setpoint_temp = df['WEST ZONE CCOIL AIR OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.westzone_air_loop_outlet_temp = df['WEST AIR LOOP OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.westzone_air_loop_outlet_setpoint_temp = df['WEST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']

        #XX self.eastzone_return_air_temp = df['EAST ZONE RETURN AIR NODE:System Node Temperature [C](TimeStep)']
        #XXself.eastzone_mixed_air_temp = df['EAST ZONE MIXED AIR NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_supply_fan_outlet_temp = df['EAST ZONE SUPPLY FAN OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_dec_outlet_temp = df['EAST ZONE DEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_dec_outlet_setpoint_temp = df['EAST ZONE DEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.eastzone_iec_outlet_temp = df['EAST ZONE IEC OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_iec_outlet_setpoint_temp = df['EAST ZONE IEC OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.eastzone_ccoil_air_outlet_temp = df['EAST ZONE CCOIL AIR OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_ccoil_air_outlet_setpoint_temp = df['EAST ZONE CCOIL AIR OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']
        self.eastzone_air_loop_outlet_temp = df['EAST AIR LOOP OUTLET NODE:System Node Temperature [C](TimeStep)']
        self.eastzone_air_loop_outlet_setpoint_temp = df['EAST AIR LOOP OUTLET NODE:System Node Setpoint Temperature [C](TimeStep)']

        # Electric power
        self.total_building_electric_demand_power = df['Whole Building:Facility Total Building Electric Demand Power [W](Hourly)']
        self.total_hvac_electric_demand_power = df['Whole Building:Facility Total HVAC Electric Demand Power [W](Hourly)']
        self.total_electric_demand_power = df['Whole Building:Facility Total Electric Demand Power [W](Hourly)']

        # Compute reward list
        self.rewards = []
        self.rewards_gaussian1 = []
        self.rewards_trapezoid1 = []
        self.rewards_gaussian2 = []
        self.rewards_trapezoid2 = []
        self.rewards_power = []
        
        for Tenv, Tz1, Tz2, PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power in zip(
                self.outdoor_temp,
                self.westzone_temp,
                self.eastzone_temp,
                self.pue,
                self.total_electric_demand_power,
                self.total_building_electric_demand_power,
                self.total_hvac_electric_demand_power):
            rew, elem = self._compute_reward([Tenv, Tz1, Tz2, PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power])
            self.rewards.append(rew)
            self.rewards_gaussian1.append(elem[0])
            self.rewards_trapezoid1.append(elem[1])
            self.rewards_gaussian2.append(elem[2])
            self.rewards_trapezoid2.append(elem[3])
            self.rewards_power.append(elem[4])
        
        # Cooling and heating setpoint for ZoneControl:Thermostat
        self.cooling_setpoint = []
        self.heating_setpoint = []
        for dt in date_time:
            self.cooling_setpoint.append(24.0)
            self.heating_setpoint.append(23.0)
        
        (self.x_pos, self.x_labels) = self.generate_x_pos_x_labels(date)

    def plot_episode(self, ep):
        self.stats_utils.plot_episode(self, ep, *['tmp'])