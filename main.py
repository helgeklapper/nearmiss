# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:52:51 2017
Python 3.x
@author: Helge Klapper
"""

import copy
import datetime
from multiprocessing import Pool, Manager
import numpy as np

from output import create_dirs, create_graphs, write_csv
from simulation import simulation, time_left


class Config:
    # Number of Environments sampled
    E = 1000

    # Number of rounds
    ROUNDS = 100

    # Number of parts/machines/divisions (columns)
    X = 16

    # Number of fail-safes (layers, rows)
    Y = 5

    # Number of agents
    N = 32

    # Average starting threshold for individuals
    START_E = 0.0

    # Errors placed on starting map
    S_THRESH = 0.5

    # Variance in individual thresholds
    THRESH_SD = 0.001

    # starting weight for organization
    NOISE = 0.5

    # Noise of signal that agents receive
    NORMAL = 0

    # Type of signal that agents receive
    ORG_WEIGHT = 0.5

    # Probability that machine (cell) becomes damaged
    PROB_E = 0.03

    # Probability that if machine is damaged, machine breaks down
    PROB_E_SD = 0.03

    # Standard deviation of latent error
    PROB_A = 0.75

    # Improvement factors when latent error detected
    IMPROVE = 0.0

    # Are agents allowed to be in same location
    # SAME_POS = False

    # Tendency to change threshold upwards and downwards
    D_ORG = 0.15
    D_UP = 0.2
    D_DOWN = 0.4

    # Organizational constraint to check errors
    ORG_CHECK = 32

    # Organizational threshold to accept signal
    ORG_THRESH = 0.5

    # What decision structure is used
    DEC_STRU = 0

    # Organizational detection capability
    ORG_DETECT = 1

    # Use divisional checks instead of overall
    MIDDLE = 1

    # When failure happens, are all errors reset?
    RESET = 0

    # For graphs
    DPI = 200


class Params:
    COLUMNS = {0: ('E', 'Environment'),
               1: ('n', 'Agents'),
               2: ('x', 'Width'),
               3: ('y', 'Height'),
               4: ('prob_e', 'Probability of potential error'),
               5: ('prob_e_sd', 'Probability of pot. error deviation'),
               6: ('prob_a', 'Probability of activated error'),
               7: ('start_e', 'Initial error rate'),
               8: ('s_thresh', 'Starting threshold'),
               9: ('thresh_sd', 'Starting threshold variance'),
               10: ('noise', 'Noise in agent signal'),
               11: ('normal', 'Noise distribution'),
               12: ('org_weight', 'Starting Org. weight'),
               13: ('org_thresh', 'Org. accept threshold'),
               14: ('d_up', 'Delta Agents Commission'),
               15: ('d_down', 'Delta Agents Omission'),
               16: ('d_org', 'Delta Org.'),
               17: ('dec_stru', 'Decision Structure'),
               18: ('org_check', 'Org. constraint'),
               19: ('reset', 'Reset after failure'),
               20: ('org_detect', 'Org. detection capability'),
               21: ('middle', 'No. of middle managers'),
               # After here output variables
               22: ('pathogens', 'Pct. of potential errors'),
               23: ('errors', 'Pct. of activated errors'),
               24: ('threshold', 'Average threshold'),
               25: ('threshold_sd', 'Threshold Std. Dev.'),
               26: ('reported', 'Pct. of agents reporting'),
               27: ('listened', 'Pct. of fields investigated'),
               28: ('repaired', 'Pct. of fields repaired'),
               29: ('omission', 'Pct. omission errors'),
               30: ('commission', 'Pct. commission errors'),
               31: ('ind_error', 'Average error rate'),
               32: ('feedback_fail', 'Pct. feedback failure'),
               33: ('feedback_omit', 'Pct. feedback omission'),
               34: ('feedback_commit', 'Pct. feedback commission'),
               35: ('org_weight', 'Org. weight'),
               36: ('org_correct', 'Overall signal correct'),
               37: ('agents_correct', 'Agent signal correct'),
               38: ('agents_percentage', 'Pct. report. workers correct'),
               39: ('near_miss', 'Near Miss'),
               40: ('near_det', 'Near Miss detected'),
               41: ('near_det_ave', 'Average near Miss detected'),
               42: ('near_det_roll', 'Near Miss detected (roll. ave.)'),
               43: ('failure', 'Failure Rate'),
               44: ('failure_roll', 'Failure Rate'),
               45: ('failure_ave', 'Average Failure Rate'),
               46: ('failure_dummy', 'Failed Organizations')
               }

    NO_ATTRIBUTES = len(COLUMNS)

    """
    Define your (two) variables and range here.
    Use number from above (table columns)
    First number defines the graphs
    Second number defines x axis
    Make sure to change values in loop as well
    ROUNDS NOT POSSIBLE AS VARIABLE
    GRAPH 3 takes care of rounds as IV
    """
    VAR_1 = 8
    VAR_2 = 20
    if VAR_2 == 2:
        Config.Y = Config.X
        Config.N = int(Config.X * Config.Y * 0.2)
        Config.ORG_CHECK = Config.N / 2

    # For integers use arange and for floats use linspace
    VAR_1_VALUES = (0.8, 0.5, 0.2)
    VAR_2_VALUES = (0.6, 0.7, 0.8, 0.9, 1.0)

    VAR_1_NAME = str(COLUMNS[VAR_1][0])
    VAR_2_NAME = str(COLUMNS[VAR_2][0])

    VAR_1_LABEL = str(COLUMNS[VAR_1][1])
    VAR_2_LABEL = str(COLUMNS[VAR_2][1])


class ArgumentStruct:
    """
    Struct to hold the Config values, which can be adjusted
    """

    def __init__(self):
        # Copy all config values
        for key, value in Config.__dict__.items():
            if not key.startswith('__'):
                self.__setattr__(key, copy.deepcopy(value))
        # Copy needed Param value
        self.NO_ATTRIBUTES = len(Params.COLUMNS)
        self.COLUMNS = Params.COLUMNS


def show_first_arguments(first_args):
    print()
    print('Environments sampled                         :', first_args.E)
    print('Number of Rounds                             :', first_args.ROUNDS)
    print('Number of Agents                             :', first_args.N)
    print('Dimensions                                   :',
          first_args.Y, first_args.X)
    print('Initial error, prob. of Error and Activation :',
          first_args.START_E, first_args.PROB_E, first_args.PROB_A)
    print('Starting Threshold and Variance              :',
          first_args.S_THRESH, first_args.THRESH_SD)
    print('Starting Org. Weight and Threshold           :',
          first_args.ORG_WEIGHT, first_args.ORG_THRESH)
    print('Downward and upward updating                 :',
          first_args.D_UP, first_args.D_DOWN)
    print('Org. updating                                :', first_args.D_ORG)
    print('Reset after failure                          :', first_args.RESET)
    print()


def get_argument_sets(results_dict):
    """
    Create iterable with argument structs, instance and a ref to the result dict
    """

    var1_name = Params.COLUMNS[Params.VAR_1][0].upper()
    var2_name = Params.COLUMNS[Params.VAR_2][0].upper()

    argument_sets = []
    instance = 0
    for var1 in Params.VAR_1_VALUES:
        for var2 in Params.VAR_2_VALUES:
            arguments = ArgumentStruct()
            setattr(arguments, var1_name, var1)
            setattr(arguments, var2_name, var2)
            argument_sets.append((arguments, instance, results_dict))
            instance += 1

            # Show the first argument set
            if instance == 1:
                show_first_arguments(arguments)

    return argument_sets


def main_loop(show_progress=False):
    np.warnings.filterwarnings('ignore')

    argument_sets = get_argument_sets(None)
    RES = np.zeros((len(argument_sets), Config.ROUNDS, Params.NO_ATTRIBUTES))
    for arguments, instance, _ in argument_sets:
        if show_progress:
            temp_number = int(
                np.floor(instance / float(len(Params.VAR_2_VALUES))))
            print(Params.VAR_1_LABEL, ':',
                  Params.VAR_1_VALUES[temp_number])
            temp_number = (instance % len(Params.VAR_2_VALUES))
            print(Params.VAR_2_LABEL, ':',
                  Params.VAR_2_VALUES[temp_number])
        RES[instance, :, :] = simulation(arguments)
        instance += 1
        if show_progress:
            print('Instance No.: ', instance)
            c_time = datetime.datetime.now().replace(microsecond=0)
            tleft = time_left(c_time, instance, time_0,
                              len(Params.VAR_1_VALUES) * len(
                                  Params.VAR_2_VALUES))
            print('Time left:', tleft)
    return RES


def wrapper(args):
    """
    Unpack the arguments, run simulation and add it to the results
    """
    np.warnings.filterwarnings('ignore')

    arguments, instance, results_dict = args
    results_dict[instance] = simulation(arguments)


def main_loop_multi():
    # Use a process-safe dictionary to hold the intermediate results
    results_dict = Manager().dict()
    argument_sets = get_argument_sets(results_dict)
    with Pool() as pool:
        pool.map(wrapper, argument_sets)

    RES = np.zeros((len(argument_sets), Config.ROUNDS, Params.NO_ATTRIBUTES))
    for instance_id in sorted(results_dict.keys()):
        result = results_dict[instance_id]
        RES[instance_id, :, :] = result
    return RES


if __name__ == "__main__":
    time_0 = datetime.datetime.now().replace(microsecond=0)

    RES = main_loop_multi()
    # RES = main_loop()
    time_1 = datetime.datetime.now().replace(microsecond=0)

    run_dir, csv_path = create_dirs(Params.VAR_1_NAME, Params.VAR_2_NAME)
    create_graphs(run_dir, RES)
    time_2 = datetime.datetime.now().replace(microsecond=0)

    write_csv(csv_path, RES)
    time_3 = datetime.datetime.now().replace(microsecond=0)

    print()
    print('Main loop duration:', time_1 - time_0)
    print('Graphs duration   :', time_2 - time_1)
    print('Finished, duration:', time_3 - time_0)