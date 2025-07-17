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
import warnings
from output import create_dirs, create_graphs, write_csv
from simulation import simulation


class Config:
    # Number of Environments sampled
    E = 1000

    # Number of rounds
    ROUNDS = 100

    # Number of parts/machines/divisions (columns)
    X = 8

    # Number of fail-safes (layers, rows)
    Y = 6

    # Number of agents
    N = 8

    # Errors placed on starting map
    START_E = 0.0

    # Selection parameter for softmax
    TAU = 0.5

    # Probability that machine (cell) becomes damaged
    PROB_E = 0.35

    # Standard deviation of latent error
    PROB_E_SD = 0.20

    # Probability that if machine is damaged, machine breaks down
    PROB_A = 1

    # Reporting error rate
    REP_ERROR = 0.0

    # Organizational constraint to check errors
    ORG_CAP = N

    # What decision structure is used
    CENTRAL = 0

    # Report delay
    CEN_DELAY = 0

    # Use divisional checks instead of overall
    MIDDLE = 1

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
               8: ('tau', 'Softmax parameter'),
               9: ('reset', 'Reset after failure'),
               10: ('org_cap', 'Manager detection capability'),
               11: ('middle', 'Divisions'),
               12: ('central', 'Centralization'),
               13: ('rep_error', 'Reporting error'),
               # After here output variables
               14: ('errors', 'Activated errors'),
               15: ('reported', 'Agents reporting'),
               16: ('inv_agent', 'Units investigated (Agents)'),
               17: ('inv_check', 'Units investigated (Capacity)'),
               18: ('repaired', 'Units repaired'),
               19: ('omission', 'False negative rate'),
               20: ('commission', 'False positive rate'),
               21: ('ind_error', 'Average false report rate'),
               22: ('agents_correct', 'Accuracy of workers'),
               23: ('agents_percentage', 'Accuracy of workers'),
               24: ('failure', 'Failure rate'),
               25: ('failure_roll', 'Failure rate (rolling)'),
               26: ('failure_ave', 'Average failure rate'),
               27: ('failure_dummy', 'Failed organizations')
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

    VAR_1 = 12
    VAR_2 = 8

    if VAR_2 == 2:
        Config.Y = Config.X
        Config.N = int(Config.X * Config.Y * 0.2)
        Config.ORG_CHECK = Config.N / 2
    elif VAR_2 == 14:
        Config.D_DOWN = Config.D_UP / 2

    # For integers use arange and for floats use linspace

    VAR_1_VALUES = [0, 1]
    VAR_2_VALUES = np.arange(0.01, 2.1, 0.5)

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
    print('Reset after failure                          :', first_args.RESET)
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
    warnings.filterwarnings('ignore')

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
    return RES


def wrapper(args):
    """
    Unpack the arguments, run simulation and add it to the results
    """
    warnings.filterwarnings('ignore')

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
    with Pool(processes=4) as pool:
        pool.map(wrapper, argument_sets)

    RES = np.zeros((len(argument_sets), Config.ROUNDS, Params.NO_ATTRIBUTES), dtype=np.float32)
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
