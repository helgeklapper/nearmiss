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
    E = 5000

    # Number of rounds
    ROUNDS = 100

    # Number of parts/machines/divisions (columns)
    X = 16

    # Number of fail-safes (layers, rows)
    Y = 8

    # Number of agents
    N = 32

    # Errors placed on starting map
    START_E = 0.0

    # Average starting threshold for individuals
    S_TEND = 0.5

    # Variance in individual thresholds
    TEND_SD = 0.001

    # Noise of signal that agents receive
    NOISE = 0.5

    # Type of signal that agents receive
    NORMAL = 0

    # starting weight for organization
    S_ORG_WEIGHT = 0.5

    # Probability that machine (cell) becomes damaged
    PROB_E = 0.03

    # Standard deviation of latent error
    PROB_E_SD = 0.00

    # Probability that if machine is damaged, machine breaks down
    PROB_A = 0.8

    # Improvement factors when latent error detected
    IMPROVE = 0.0

    # Are agents allowed to be in same location
    # SAME_POS = False

    # Tendency to change threshold upwards and downwards
    D_ORG = 0.15
    D_UP = 0.4
    D_DOWN = 0.2

    # Organizational constraint to check errors
    ORG_CHECK = 6

    # Organizational constraint to check errors
    ORG_CHECK_CHANGE = 0

    # Organizational threshold to accept signal
    ORG_THRESH = 0.5

    # What decision structure is used
    DEC_STRU = 0

    # Organizational detection capability
    ORG_DETECT = 1

    # Use divisional checks instead of overall
    MIDDLE = 1

    # Interdependence of subsystems
    LINEAR = 0

    # Coupling
    COUPLING = 1

    # When failure happens, are all errors reset?
    RESET = 0

    # For graphs
    DPI = 300


class Params:
    COLUMNS = {0: ('E', 'Environment'),
               1: ('n', 'Agents'),
               2: ('x', 'Width'),
               3: ('y', 'Height'),
               4: ('prob_e', 'Probability of potential error'),
               5: ('prob_e_sd', 'Probability of pot. error deviation'),
               6: ('prob_a', 'Probability of activated error'),
               7: ('start_e', 'Initial error rate'),
               8: ('s_tend', 'Initial reporting climate'),
               9: ('thresh_sd', 'Starting threshold variance'),
               10: ('noise', 'Noise in agent signal'),
               11: ('normal', 'Noise distribution'),
               12: ('s_org_weight', 'Weight on worker reports'),
               13: ('org_thresh', 'Org. accept threshold'),
               14: ('d_up', 'Omission Feedback'),
               15: ('d_down', 'Commission Feedback'),
               16: ('d_org', 'Organizational Reactivity'),
               17: ('dec_stru', 'Decision Structure'),
               18: ('org_check', 'Org. constraint'),
               19: ('org_check_change', 'Variable constraint'),
               20: ('reset', 'Reset after failure'),
               21: ('org_detect', 'Manager detection capability'),
               22: ('middle', 'Divisions'),
               23: ('linear', 'Independence'),
               24: ('coupling', 'Tight Coupling'),
               # After here output variables
               25: ('pathogens', 'Potential errors'),
               26: ('errors', 'Activated errors'),
               27: ('tend', 'Reporting climate'),
               28: ('tend_sd', 'Tendency Std. Dev.'),
               29: ('reported', 'Agents reporting'),
               30: ('inv_agent', 'Units investigated (Agents)'),
               31: ('inv_check', 'Units investigated (Capacity)'),
               32: ('repaired', 'Units repaired'),
               33: ('omission', 'Omission errors'),
               34: ('commission', 'Commission errors'),
               35: ('ind_error', 'Average error rate'),
               36: ('feedback_fail', 'Feedback failure'),
               37: ('feedback_omit', 'Feedback omission'),
               38: ('feedback_commit', 'Feedback commission'),
               39: ('org_check', 'Org. investigation capability'),
               40: ('org_weight', 'Weight on worker reports'),
               41: ('org_correct', 'Overall signal correct'),
               42: ('agents_correct', 'Accuracy of workers'),
               43: ('agents_percentage', 'Accuracy of workers'),
               44: ('near_miss', 'Near failure rate'),
               45: ('near_det', 'Near failure detected'),
               46: ('near_det_ave', 'Average near failure detected'),
               47: ('near_det_roll', 'Near failure detected'),
               48: ('failure', 'Failure rate'),
               49: ('failure_roll', 'Failure rate'),
               50: ('failure_ave', 'Average failure rate'),
               51: ('failure_dummy', 'Failed organizations')
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

    VAR_1 = 4
    VAR_2 = 8

    if VAR_2 == 2:
        Config.Y = Config.X
        Config.N = int(Config.X * Config.Y * 0.2)
        Config.ORG_CHECK = Config.N / 2
    elif VAR_2 == 14:
        Config.D_DOWN = Config.D_UP / 2

    # For integers use arange and for floats use linspace

    VAR_1_VALUES = [0.03]
    VAR_2_VALUES = np.arange(0.1, 1, 0.1)

    # np.arange(16,95,16)
    # np.arange(0.1, 1, 0.4)
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
          first_args.S_TEND, first_args.TEND_SD)
    print('Starting Org. Weight and Threshold           :',
          first_args.S_ORG_WEIGHT, first_args.ORG_THRESH)
    print('Downward and upward updating                 :',
          first_args.D_UP, first_args.D_DOWN)
    print('Org. updating                                :', first_args.D_ORG)
    print('Reset after failure                          :', first_args.RESET)
    print('Indepence                                    :', first_args.LINEAR)
    print()


def get_argument_sets(results_dict):
    """
    Create iterable with argument structs,
    instance and a ref to the result dict
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


def main_loop(show_progress=True):
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
    print('Instance Wrapper No.: ', instance)
    print('Time: ', datetime.datetime.now().replace(microsecond=0))


def main_loop_multi():
    # Use a process-safe dictionary to hold the intermediate results
    results_dict = Manager().dict()
    argument_sets = get_argument_sets(results_dict)
    # processes=2
    instances = len(Params.VAR_1_VALUES) * len(Params.VAR_2_VALUES)
    print('Time: ', datetime.datetime.now().replace(microsecond=0))
    print('Instances', instances)
    with Pool(processes=6) as pool:
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
