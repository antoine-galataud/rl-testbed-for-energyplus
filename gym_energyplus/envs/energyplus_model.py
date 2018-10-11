# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from abc import ABCMeta, abstractmethod
import os, sys, time
from scipy.special import expit
import numpy as np
from datetime import datetime, timedelta
from glob import glob
from matplotlib.widgets import Slider, Button, RadioButtons
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import json
from gym_energyplus.envs.energyplus_episode_stats import EpisodeStatsUtils

class EnergyPlusModel(metaclass=ABCMeta):

    def __init__(self,
                 model_file,
                 log_dir=None,
                 verbose=False):
        self.log_dir = log_dir
        self.model_basename = os.path.splitext(os.path.basename(model_file))[0]
        self.setup_spaces()
        self.action = 0.5 * (self.action_space.low + self.action_space.high)
        self.action_prev = self.action
        self.raw_state = None
        self.verbose = verbose
        self.timestamp_csv = None
        self.sl_episode = None

        # Progress data
        self.num_episodes = 0
        self.num_episodes_last = 0

        self.reward = None
        self.reward_mean = None

        self.stats_utils = EpisodeStatsUtils()

    def reset(self):
        pass

    # Parse date/time format from EnergyPlus and return datetime object with correction for 24:00 case
    def _parse_datetime(self, dstr):
        # ' MM/DD  HH:MM:SS' or 'MM/DD  HH:MM:SS'
        # Dirty hack
        if dstr[0] != ' ':
            dstr = ' ' + dstr
        #year = 2017
        year = 2013 # for CHICAGO_IL_USA TMY2-94846
        month = int(dstr[1:3])
        day = int(dstr[4:6])
        hour = int(dstr[8:10])
        minute = int(dstr[11:13])
        sec = 0
        msec = 0
        if hour == 24:
            hour = 0
            dt = datetime(year, month, day, hour, minute, sec, msec) + timedelta(days=1)
        else:
            dt = datetime(year, month, day, hour, minute, sec, msec)
        return dt

    # Convert list of date/time string to list of datetime objects
    def _convert_datetime24(self, dates):
        # ' MM/DD  HH:MM:SS'
        dates_new = []
        for d in dates:
            #year = 2017
            #month = int(d[1:3])
            #day = int(d[4:6])
            #hour = int(d[8:10])
            #minute = int(d[11:13])
            #sec = 0
            #msec = 0
            #if hour == 24:
            #    hour = 0
            #    d_new = datetime(year, month, day, hour, minute, sec, msec) + dt.timedelta(days=1)
            #else:
            #    d_new = datetime(year, month, day, hour, minute, sec, msec)
            #dates_new.append(d_new)
            dates_new.append(self._parse_datetime(d))
        return dates_new

    # Generate x_pos and x_labels
    def generate_x_pos_x_labels(self, dates):
        time_delta  = self._parse_datetime(dates[1]) - self._parse_datetime(dates[0])
        x_pos = []
        x_labels = []
        for i, d in enumerate(dates):
            dt = self._parse_datetime(d) - time_delta
            if dt.hour == 0 and dt.minute == 0:
                x_pos.append(i)
                x_labels.append(dt.strftime('%m/%d'))
        return x_pos, x_labels

    def set_action(self, normalized_action):
        # In TPRO/PPO1/PPO2 in baseline, action seems to be normalized to [-1.0, 1.0].
        # So it must be scaled back into action_space by the environment.
        self.action_prev = self.action
        self.action = self.action_space.low + (normalized_action + 1.) * 0.5 * (self.action_space.high - self.action_space.low)
        self.action = np.clip(self.action, self.action_space.low, self.action_space.high)

    def get_state(self):
        return self.format_state(self.raw_state)

    @abstractmethod
    def setup_spaces(self): pass

    # Need to handle the case that raw_state is None
    @abstractmethod
    def set_raw_state(self, raw_state): pass

    @abstractmethod
    def compute_reward(self): pass

    @abstractmethod
    def format_state(self, raw_state): pass

    @abstractmethod
    def read_episode(self, ep): pass

    @abstractmethod
    def plot_episode(self, ep): pass

    #--------------------------------------------------
    # Plotting staffs follow
    #--------------------------------------------------
    def plot(self, log_dir='', csv_file='', **kwargs):
        if log_dir is not '':
            if not os.path.isdir(log_dir):
                print('energyplus_model.plot: {} is not a directory'.format(log_dir))
                return
            print('energyplus_plot.plot log={}'.format(log_dir))
            self.log_dir = log_dir
            self.show_progress()
        else:
            if not os.path.isfile(csv_file):
                print('energyplus_model.plot: {} is not a file'.format(csv_file))
                return
            print('energyplus_model.plot csv={}'.format(csv_file))
            self.read_episode(csv_file)
            plt.rcdefaults()
            plt.rcParams['font.size'] = 6
            plt.rcParams['lines.linewidth'] = 1.0
            plt.rcParams['legend.loc'] = 'lower right'
            self.fig = plt.figure(1, figsize=(16, 10))
            self.plot_episode(csv_file)
            plt.tight_layout()
            plt.show()

    # Show convergence
    def show_progress(self):
        self.monitor_file = self.log_dir + '/monitor.csv'

        # Read progress file
        if not self.read_monitor_file():
            print('Progress data is missing')
            sys.exit(1)

        # Initialize graph
        plt.rcdefaults()
        plt.rcParams['font.size'] = 6
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams['legend.loc'] = 'lower right'

        self.fig = plt.figure(1, figsize=(16, 10))

        # Show widgets
        axcolor = 'lightgoldenrodyellow'
        self.axprogress = self.fig.add_axes([0.15, 0.10, 0.70, 0.15], facecolor=axcolor)
        self.axslider = self.fig.add_axes([0.15, 0.04, 0.70, 0.02], facecolor=axcolor)
        axfirst = self.fig.add_axes([0.15, 0.01, 0.03, 0.02])
        axlast = self.fig.add_axes([0.82, 0.01, 0.03, 0.02])
        axprev = self.fig.add_axes([0.46, 0.01, 0.03, 0.02])
        axnext = self.fig.add_axes([0.51, 0.01, 0.03, 0.02])

        # Slider is drawn in plot_progress()

        # First/Last button
        self.button_first = Button(axfirst, 'First', color=axcolor, hovercolor='0.975')
        self.button_first.on_clicked(self.first_episode_num)
        self.button_last = Button(axlast, 'Last', color=axcolor, hovercolor='0.975')
        self.button_last.on_clicked(self.last_episode_num)

        # Next/Prev button
        self.button_prev = Button(axprev, 'Prev', color=axcolor, hovercolor='0.975')
        self.button_prev.on_clicked(self.prev_episode_num)
        self.button_next = Button(axnext, 'Next', color=axcolor, hovercolor='0.975')
        self.button_next.on_clicked(self.next_episode_num)

        # Timer
        self.timer = self.fig.canvas.new_timer(interval=1000)
        self.timer.add_callback(self.check_update)
        self.timer.start()

        # Progress data
        self.axprogress.set_xmargin(0)
        self.axprogress.set_xlabel('Episodes')
        self.axprogress.set_ylabel('Reward')
        self.axprogress.grid(True)
        self.plot_progress()

        # Plot latest episode
        self.update_episode(self.num_episodes - 1)

        plt.show()

    def check_update(self):
        if self.read_monitor_file():
            self.plot_progress()

    def plot_progress(self):
        # Redraw all lines
        self.axprogress.lines = []
        self.axprogress.plot(self.reward, color='#1f77b4', label='Reward')
        #self.axprogress.plot(self.reward_mean, color='#ff7f0e', label='Reward (average)')
        self.axprogress.legend()
        # Redraw slider
        if self.sl_episode is None or int(round(self.sl_episode.val)) == self.num_episodes - 2:
            cur_ep = self.num_episodes - 1
        else:
            cur_ep = int(round(self.sl_episode.val))
        self.axslider.clear()
        #self.sl_episode = Slider(self.axslider, 'Episode (0..{})'.format(self.num_episodes - 1), 0, self.num_episodes - 1, valinit=self.num_episodes - 1, valfmt='%6.0f')
        self.sl_episode = Slider(self.axslider, 'Episode (0..{})'.format(self.num_episodes - 1), 0, self.num_episodes - 1, valinit=cur_ep, valfmt='%6.0f')
        self.sl_episode.on_changed(self.set_episode_num)

    def read_monitor_file(self):
        # For the very first call, Wait until monitor.csv is created
        if self.timestamp_csv is None:
            while not os.path.isfile(self.monitor_file):
                time.sleep(1)
            self.timestamp_csv = os.stat(self.monitor_file).st_mtime - 1 # '-1' is a hack to prevent losing the first set of data

        num_ep = 0
        ts = os.stat(self.monitor_file).st_mtime
        if ts > self.timestamp_csv:
            # Monitor file is updated.
            self.timestamp_csv = ts
            f = open(self.monitor_file)
            firstline = f.readline()
            assert firstline.startswith('#')
            metadata = json.loads(firstline[1:])
            assert metadata['env_id'] == "EnergyPlus-v0"
            assert set(metadata.keys()) == {'env_id', 't_start'},  "Incorrect keys in monitor metadata"
            df = pd.read_csv(f, index_col=None)
            assert set(df.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
            f.close()

            self.reward = []
            self.reward_mean = []
            self.episode_dirs = []
            for rew, len, time in zip(df['r'], df['l'], df['t']):
                self.reward.append(rew / len)
                self.reward_mean.append(rew / len)
                self.episode_dirs.append(self.log_dir + '/output/episode-{:08d}'.format(self.num_episodes))
                self.num_episodes += 1
            if self.num_episodes > self.num_episodes_last:
                self.num_episodes_last = self.num_episodes
                return True
        else:
            return False

    def update_episode(self, ep):
        self.plot_episode(ep)

    def set_episode_num(self, val):
        ep = int(round(self.sl_episode.val))
        self.update_episode(ep)

    def first_episode_num(self, val):
        self.sl_episode.set_val(0)

    def last_episode_num(self, val):
        self.sl_episode.set_val(self.num_episodes - 1)

    def prev_episode_num(self, val):
        ep = int(round(self.sl_episode.val))
        if ep > 0:
            ep -= 1
            self.sl_episode.set_val(ep)

    def next_episode_num(self, val):
        ep = int(round(self.sl_episode.val))
        if ep < self.num_episodes - 1:
            ep += 1
            self.sl_episode.set_val(ep)

    def get_episode_list(self, log_dir='', csv_file=''):
        if (log_dir is not '' and csv_file is not '') or (log_dir is '' and csv_file is ''):
            print('Either one of log_dir or csv_file must be specified')
            quit()
        if log_dir is not '':
            if not os.path.isdir(log_dir):
                print('energyplus_model.dump: {} is not a directory'.format(log_dir))
                return
            print('energyplus_plot.dump: log={}'.format(log_dir))
            #self.log_dir = log_dir

            # Make a list of all episodes
            # Note: Somethimes csv file is missing in the episode directories
            # We accept gziped csv file also.
            csv_list = glob(log_dir + '/output/episode-????????/eplusout.csv') \
                       + glob(log_dir + '/output/episode-????????/eplusout.csv.gz')
            self.episode_dirs = list(set([os.path.dirname(i) for i in csv_list]))
            self.episode_dirs.sort()
            self.num_episodes = len(self.episode_dirs)
        else: #csv_file != ''
            self.episode_dirs = [ os.path.dirname(csv_file) ]
            self.num_episodes = len(self.episode_dirs)

    #--------------------------------------------------
    # Dump timesteps
    #--------------------------------------------------
    def dump_timesteps(self, log_dir='', csv_file='', **kwargs):
        def rolling_mean(data, size, que):
            out = []
            for d in data:
                que.append(d)
                if len(que) > size:
                    que.pop(0)
                out.append(sum(que) / len(que))
            return out
        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        print('{} episodes'.format(self.num_episodes))
        with open('dump_timesteps.csv', mode='w') as f:
            tot_num_rec = 0
            f.write('Sequence,Episode,Sequence in episode,Reward,tz1,tz2,power,Reward(avg1000)\n')
            que = []
            for ep in range(self.num_episodes):
                print('Episode {}'.format(ep))
                self.read_episode(ep)
                rewards_avg = rolling_mean(self.rewards, 1000, que)
                ep_num_rec = 0
                for rew, tz1, tz2, pow, rew_avg in zip(
                        self.rewards,
                        self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)'],
                        self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)'],
                        self.df['Whole Building:Facility Total Electric Demand Power [W](Hourly)'],
                        rewards_avg):
                    f.write('{},{},{},{},{},{},{},{}\n'.format(tot_num_rec, ep, ep_num_rec, rew, tz1, tz2, pow, rew_avg))
                    tot_num_rec += 1
                    ep_num_rec += 1

    #--------------------------------------------------
    # Dump episodes
    #--------------------------------------------------
    def dump_episodes(self, log_dir='', csv_file='', **kwargs):
        self.get_episode_list(log_dir=log_dir, csv_file=csv_file)
        print('{} episodes'.format(self.num_episodes))
        with open('dump_episodes.dat', mode='w') as f:
            tot_num_rec = 0
            f.write('#Test Ave1  Min1  Max1 STD1  Ave2  Min2  Max2 STD2   Rew     Power [22,25]1 [22,25]2  Ep\n')
            for ep in range(self.num_episodes):
                print('Episode {}'.format(ep))
                self.read_episode(ep)
                Temp1 = self.df['WEST ZONE:Zone Air Temperature [C](TimeStep)']
                Temp2 = self.df['EAST ZONE:Zone Air Temperature [C](TimeStep)']
                Ave1, Min1, Max1, STD1 = self.get_statistics(Temp1)
                Ave2, Min2, Max2, STD2 = self.get_statistics(Temp2)
                In22_24_1 = np.sum((Temp1 >= 22.0) & (Temp1 <= 25.0)) / len(Temp1)
                In22_25_2 = np.sum((Temp2 >= 22.0) & (Temp2 <= 25.0)) / len(Temp2)
                Rew, _, _, _ = self.get_statistics(self.rewards)
                Power, _, _, _ = self.get_statistics(self.df['Whole Building:Facility Total Electric Demand Power [W](Hourly)'])
                
                f.write('"{}" {:5.2f} {:5.2f} {:5.2f} {:4.2f} {:5.2f} {:5.2f} {:5.2f} {:4.2f} {:5.2f} {:9.2f} {:8.3%} {:8.3%} {:3d}\n'.format(self.weather_key, Ave1, Min1, Max1, STD1, Ave2,  Min2, Max2, STD2, Rew, Power, In22_25_1, In22_25_2, ep))

