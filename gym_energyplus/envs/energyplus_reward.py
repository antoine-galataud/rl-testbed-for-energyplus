import numpy as np
import os

"""
Reward computation functions
"""
class RewardUtils:

    def compute_reward(self, env):
        self.raw_state = env.raw_state
        self.action = env.action
        self.action_prev = env.action_prev
        rew, _ = self._compute_reward()
        return rew

    def _compute_reward(self, raw_state = None):
        #return self.compute_reward_center23_5_gaussian1_0_trapezoid1_0_pue0_0(raw_state)
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
        
    def compute_reward_center23_5_gaussian1_0_trapezoid1_0_pue0_0(self, raw_state = None): # gaussian/trapezoid, PUE
        return self.compute_reward_common(
            temperature_center = 23.5,
            temperature_tolerance = 0.5,
            temperature_gaussian_weight = 1.0,
            temperature_gaussian_sharpness = 0.5,
            temperature_trapezoid_weight = 1.0,
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
            print('compute_reward: Tenv={:7.3f}, Tz1={:7.3f}, Tz2={:7.3f}, PUE={:7.3f}, Whole_Powerd2={:8.1f}, ITE_Power={:8.1f}, HVAC_Power={:8.1f}'.format(Tenv, Tz1, Tz2, PUE, Whole_Building_Power, IT_Equip_Power, Whole_HVAC_Power))
            if len(self.actions) == 4:
                print('Act1={:7.3f}, Act2={:7.3f}, Act3={:7.3f}, Act4={:7.3f}'.format(self.action[0], self.action[1], self.action[2], self.action[3]))
            else:
                print('Act1={:7.3f}, Act2={:7.3f}'.format(self.action[0], self.action[1]))
        
        return rew, (rew_temp_gaussian1, rew_temp_trapezoid1, rew_temp_gaussian2, rew_temp_trapezoid2, rew_Whole_Building_Power)