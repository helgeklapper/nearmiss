# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:52:51 2017
Python 3.x
@author: Helge Klapper
"""
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="ticks", context="talk", )
sns.set_palette(sns.color_palette("Blues_d"))


def graph1(run_dir, variable_num, res, dpi, round_no, colors="Blues"):
    """Creates a graph for defined variable"""
    from main import Config, Params

    sns.set_palette(sns.color_palette(colors))
    params = {
        'axes.labelsize': 12,
        'font.family': 'Sans-Serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1,
        'axes.linewidth': 2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'text.usetex': False,
        'figure.figsize': [5.0, 5.0],
        'legend.loc': 'best'
    }
    plt.rcParams.update(params)

    plt.figure(dpi=dpi)
    plt.axes(frameon=0)
    plt.grid()
    ax = plt.subplot(111)
    plt.ylabel(str(Params.COLUMNS[variable_num][1]))
    plt.xlabel(Params.VAR_2_LABEL)
    plt.locator_params(axis='y', nticks=5)
    linestyles = ['-', '--']
    markers = ['v', '^', 'o', 's']

    line_no = 0
    for x in Params.VAR_1_VALUES:
        style = linestyles[(line_no % len(linestyles))]
        marker = markers[(line_no % len(markers))]
        line_no += 1
        # print(round_no)
        # print(VAR_1)
        M = np.extract(res[:, round_no - 1, Params.VAR_1] == x,
                       res[:, round_no - 1, Params.VAR_2])
        Z = np.extract(res[:, round_no - 1, Params.VAR_1] == x,
                       res[:, round_no - 1, variable_num])
        if Params.VAR_2 == 1:
            plt.xlabel("Ratio agents to units")
            M = np.divide(M, (Config.X * Config.Y))
        elif Params.VAR_2 == 18:
            plt.xlabel("Ratio capacity to agents")
            M = np.divide(M, Config.N)
        if Params.VAR_1 == 13 and x == 0:
            ax.plot(M, Z, label='Individual reporting', linestyle=style,
                    marker='o', markevery=2)
        elif Params.VAR_1 == 13 and x == 1:
            ax.plot(M, Z, label='Majority', linestyle=style,
                    marker=marker, markevery=2)
        elif Params.VAR_1 == 13 and x == 2:
            ax.plot(M, Z, label='Consensus', linestyle=style,
                    marker='s', markevery=2)
        elif Params.VAR_1 == 8 and x <= 0.3:
            ax.plot(M, Z, label='Low', linestyle=style,
                    marker='^', markevery=2)
        elif Params.VAR_1 == 8 and x < 0.7 and x > 0.3:
            ax.plot(M, Z, label='Moderate', linestyle=style,
                    marker='o', markevery=2)
        elif Params.VAR_1 == 8 and x >= 0.7:
            ax.plot(M, Z, label='High', linestyle=style,
                    marker='v', markevery=2)
        else:
            ax.plot(M, Z, label=str(x), linestyle=style,
                    marker=marker, markevery=2)

    box = ax.get_position()

    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.yaxis.grid(which="major", color='lightgray', linewidth=1, marker='*',
                  rasterized=True, markeredgecolor='white')

    # Put a legend to the right of the current axis
    if len(Params.VAR_1_VALUES) > 1:
        lgd = ax.legend(loc='best', title=Params.VAR_1_LABEL, frameon=True,
                        fancybox=True, framealpha=0.75)
    plt.tight_layout()

    name = "%s%s_%s_%s_round_%03d" % (variable_num,
                                      Params.COLUMNS[Params.VAR_1][0],
                                      Params.COLUMNS[Params.VAR_2][0],
                                      Params.COLUMNS[variable_num][0],
                                      round_no)

    graph_name = os.path.join(run_dir, 'png', name + '.png')
    if len(Params.VAR_1_VALUES) > 1:
        plt.savefig(graph_name, format='png', bbox_extra_artists=[lgd])
    else:
        plt.savefig(graph_name, format='png')
    """
    graph_name = os.path.join(run_dir, 'svg', name + '.svg')
    if len(Params.VAR_1_VALUES) > 1:
        plt.savefig(graph_name, format='svg', bbox_extra_artists=[lgd])
    else:
        plt.savefig(graph_name, format='svg')
    """
    plt.close()


def graph2(run_dir, var_1, results, res, dpi, round_no,
           var_2_ind=0, colors="Greens_d"):
    """Creates a graph for defined variable"""
    from main import Config, Params

    # print(results)
    sns.set_palette(sns.color_palette(colors))
    params = {
        'axes.labelsize': 12,
        'font.family': 'Sans-Serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1,
        'axes.linewidth': 2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'text.usetex': False,
        'figure.figsize': [5.0, 5.0],
        'legend.loc': 'best'
    }
    plt.rcParams.update(params)

    plt.figure(dpi=dpi)
    plt.axes(frameon=0)
    plt.grid()
    ax = plt.subplot(111)
    plt.xlabel('Rounds')
    plt.locator_params(axis='y', nticks=5)
    linestyles = ['-', '--']
    markers = ['v', '^', 'o', 's']

    plt.ylabel(results)
    if results == 'Errors':
        values = (22, 23)
    elif results == 'Reaction':
        values = (26, 27, 28)
    elif results == 'Error Rate':
        values = (29, 30)
    elif results == 'Feedback Omission':
        values = (29, 33)
    elif results == 'Feedback Commission':
        values = (30, 34)
    elif results == 'Near Miss Detection':
        values = (39, 40)

    line_no = 0
    for x in values:
        if x == 22:
            labels = 'Latent errors'
        elif x == 23:
            labels = 'Activated errors'
        else:
            labels = Params.COLUMNS[x][1]
        style = linestyles[(line_no % len(linestyles))]
        marker = markers[(line_no % len(markers))]
        line_no += 1
        # print(round_no)
        # print(VAR_1)
        X = np.arange(1, Config.ROUNDS + 1)
        # print('X\n',X)
        # print('Var 2 index: ', var_2_ind)
        var_2 = Params.VAR_2_VALUES[int(var_2_ind)]
        # rel_instance = line_no * len(VAR_2_VALUES) + var_2
        # Z = res[rel_instance, :, x]
        Z = np.extract((res[:, :, Params.VAR_1] == var_1) &
                       (res[:, :, Params.VAR_2] == var_2),
                       res[:, :, x])
        if results == 'Signal':
            X = np.extract(res[:, Config.ROUNDS - 1, Params.VAR_1] == var_2,
                           res[:, Config.ROUNDS - 1, Params.VAR_2])
            Z = np.extract(res[:, Config.ROUNDS - 1, Params.VAR_1] == var_2,
                           res[:, Config.ROUNDS - 1, x])
        # print('Z\n',Z)
        ax.plot(X, Z, label=labels, linestyle=style,
                marker=marker, markevery=2)

    box = ax.get_position()

    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.yaxis.grid(which="major", color='lightgray', linewidth=1, marker='*',
                  rasterized=True, markeredgecolor='white')

    # Put a legend to the right of the current axis
    lgd = ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.75)
    plt.tight_layout()

    name = "%s_%s_%s_%s_round_%03d" % (
        Params.VAR_1_NAME, var_1, Params.VAR_2_NAME, var_2, round_no)

    os.makedirs(os.path.join(run_dir, 'png', results), exist_ok=True)
    graph_name = os.path.join(run_dir, 'png', results, name + '.png')
    plt.savefig(graph_name, format='png', bbox_extra_artists=[lgd])
    """
    os.makedirs(os.path.join(run_dir, 'svg', results), exist_ok=True)
    graph_name = os.path.join(run_dir, 'svg', results, name + '.svg')
    plt.savefig(graph_name, format='svg', bbox_extra_artists=[lgd])
    """
    plt.close()


def graph3(run_dir, variable_num, res, dpi, var_2=0, colors="Blues_d"):
    """Creates a round based graph for defined variable
    fixed number is the value of VAR_2 taken for graph"""
    from main import Config, Params

    sns.set_palette(sns.color_palette(colors))
    params = {
        'axes.labelsize': 12,
        'font.family': 'Sans-Serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1,
        'axes.linewidth': 2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'text.usetex': False,
        'figure.figsize': [5.0, 5.0],
        'legend.loc': 'best'
    }
    plt.rcParams.update(params)
    plt.figure(dpi=dpi)
    plt.axes(frameon=0)
    plt.grid()
    ax = plt.subplot(111)
    plt.ylabel(str(Params.COLUMNS[variable_num][1]))
    plt.xlabel('Rounds')
    plt.locator_params(axis='y', nticks=5)
    linestyles = ['-', '--']
    markers = ['v', '^', 'o', 's']

    line_no = 0
    for x in Params.VAR_1_VALUES:
        """X-Axis: rounds"""
        style = linestyles[(line_no % len(linestyles))]
        marker = markers[(line_no % len(markers))]
        rel_instance = line_no * len(Params.VAR_2_VALUES) + var_2
        line_no += 1
        # print(RESULTS.shape)
        X = np.arange(1, Config.ROUNDS + 1)
        # print('X', X)
        Z = res[rel_instance, :, variable_num]
        # print('Z', Z)
        if Params.VAR_1 == 13 and x == 0:
            ax.plot(X, Z, label='Individual reporting', linestyle=style,
                    marker='o', markevery=0.26)
        elif Params.VAR_1 == 13 and x == 1:
            ax.plot(X, Z, label='Majority', linestyle=style,
                    marker=marker, markevery=0.26)
        elif Params.VAR_1 == 13 and x == 2:
            ax.plot(X, Z, label='Consensus', linestyle=style,
                    marker='s', markevery=0.26)
        elif Params.VAR_1 == 8 and x <= 0.3:
            ax.plot(X, Z, label='Low', linestyle=style,
                    marker='^', markevery=(9, 10))
        elif Params.VAR_1 == 8 and x < 0.7 and x > 0.3:
            ax.plot(X, Z, label='Moderate', linestyle=style,
                    marker='o', markevery=(9, 10))
        elif Params.VAR_1 == 8 and x >= 0.7:
            ax.plot(X, Z, label='High', linestyle=style,
                    marker='v', markevery=(9, 10))
        else:
            ax.plot(X, Z, label=str(x), linestyle=style, marker=marker,
                    markevery=(9, 10))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.yaxis.grid(which="major", color='lightgray', linewidth=1, marker='*',
                  rasterized=True, markeredgecolor='white')

    # Put a legend to the right of the current axis
    if len(Params.VAR_1_VALUES) > 1:
        lgd = ax.legend(loc='best', title=Params.VAR_1_LABEL, frameon=True,
                        fancybox=True, framealpha=0.75)
    plt.tight_layout()

    name = "Rounds_%s%s_%s_%s" % (variable_num, Params.COLUMNS[Params.VAR_1][0],
                                  var_2, Params.COLUMNS[variable_num][0])

    os.makedirs(os.path.join(run_dir, 'png', str(Params.VAR_2_VALUES[var_2])),
                exist_ok=True)
    graph_name = \
        os.path.join(run_dir, 'png', str(Params.VAR_2_VALUES[var_2]),
                     name + '.png')
    if len(Params.VAR_1_VALUES) > 1:
        plt.savefig(graph_name, format='png', bbox_extra_artists=[lgd])
    else:
        plt.savefig(graph_name, format='png')
    """
    os.makedirs(os.path.join(run_dir, 'svg', str(Params.VAR_2_VALUES[var_2])),
                exist_ok=True)
    graph_name = \
        os.path.join(run_dir, 'svg', str(Params.VAR_2_VALUES[var_2]),
                     name + '.svg')
    if len(Params.VAR_1_VALUES) > 1:
        plt.savefig(graph_name, format='svg', bbox_extra_artists=[lgd])
    else:
        plt.savefig(graph_name, format='svg')
    """

    # plt.show()
    plt.close()


def graph4(run_dir, var_1, results, res, dpi, round_no, colors="Blues_d"):
    """Creates a graph for defined variable"""
    from main import Params

    sns.set_palette(sns.color_palette(colors))
    params = {
        'axes.labelsize': 12,
        'font.family': 'Sans-Serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 1,
        'axes.linewidth': 2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'text.usetex': False,
        'figure.figsize': [5.0, 5.0],
        'legend.loc': 'best'
    }
    plt.rcParams.update(params)

    plt.figure(dpi=dpi)
    plt.axes(frameon=0)
    plt.grid()
    ax = plt.subplot(111)
    plt.ylabel(results)
    plt.xlabel(Params.VAR_2_LABEL)
    plt.locator_params(axis='y', nticks=5)
    linestyles = ['-', '--']
    markers = ['v', '^', 'o', 's']

    if results == 'Signal':
        values = (36, 38)
        plt.ylabel("Accuracy")
        if Params.VAR_2 == 6:
            plt.xlabel("Ratio latent error")
    elif results == 'Signal2':
        values = (36, 31)
        plt.ylabel("Accuracy")
    elif results == 'Failure':
        values = (39, 43)
        plt.ylabel("Ratio")
        if Params.VAR_2 == 6:
            plt.xlabel("Ratio latent error")

    line_no = 0
    for x in values:
        style = linestyles[(line_no % len(linestyles))]
        marker = markers[(line_no % len(markers))]
        labels = Params.COLUMNS[x][1]
        line_no += 1
        # print(round_no)
        # print(VAR_1)
        X = np.extract(res[:, round_no - 1, Params.VAR_1] == var_1,
                       res[:, round_no - 1, Params.VAR_2])
        Z = np.extract(res[:, round_no - 1, Params.VAR_1] == var_1,
                       res[:, round_no - 1, x])
        if x == 27:
            Z = 1 - Z
        ax.plot(X, Z, label=labels, linestyle=style, marker=marker, markevery=2)

    box = ax.get_position()

    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.yaxis.grid(which="major", color='lightgray', linewidth=1, marker='*',
                  rasterized=True, markeredgecolor='white')

    # Put a legend to the right of the current axis
    lgd = ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.75)
    plt.tight_layout()

    name = "%s_%s_%s_round_%03d" % (
        Params.VAR_1_NAME, var_1, Params.VAR_2_NAME, round_no)

    os.makedirs(os.path.join(run_dir, 'png', results), exist_ok=True)
    graph_name = os.path.join(run_dir, 'png', results, name + '.png')
    plt.savefig(graph_name, format='png', bbox_extra_artists=[lgd])
    """
    os.makedirs(os.path.join(run_dir, 'svg', results), exist_ok=True)
    graph_name = os.path.join(run_dir, 'svg', results, name + '.svg')
    plt.savefig(graph_name, format='svg', bbox_extra_artists=[lgd])
    """
    plt.close()


def create_dirs(var_1_name, var_2_name):
    """
    This creates a directrory for today if needed, and a directory for this run
    :return: the path for this run and the path of the CSV file
    """
    today = datetime.date.today().strftime("%Y%m%d")
    os.makedirs(today, exist_ok=True)
    run = 1
    run_dir = "Sim_{0}_{1}_run{2}".format(var_1_name, var_2_name, run)
    while os.path.exists(os.path.join(today, run_dir)) is True:
        run += 1
        run_dir = "Sim_{0}_{1}_run{2}".format(var_1_name, var_2_name, run)
    os.makedirs(os.path.join(today, run_dir), exist_ok=True)
    os.makedirs(os.path.join(today, run_dir, 'png'), exist_ok=True)
    os.makedirs(os.path.join(today, run_dir, 'svg'), exist_ok=True)

    csv_name = "%s_%s_%s.csv" % (today, var_1_name, var_2_name)

    return os.path.join(today, run_dir), os.path.join(today, run_dir, csv_name)


def create_graphs(run_dir, RES):
    from main import Config, Params

    if len(Params.VAR_2_VALUES) > 1:
        for number in range(22, 47):
            graph1(run_dir, number, RES, Config.DPI, 1, 'Greens_d')
            if Config.ROUNDS > 9:
                graph1(run_dir, number, RES, Config.DPI, 10, 'Reds_d')
            if Config.ROUNDS > 10:
                graph1(run_dir, number, RES, Config.DPI, 20, 'Reds_d')
            if Config.ROUNDS > 20:
                graph1(run_dir, number, RES, Config.DPI, 50, 'Blues_d')
            if Config.ROUNDS > 50:
                graph1(run_dir, number, RES, Config.DPI, Config.ROUNDS,
                       'Blues_d')
        for values in Params.VAR_1_VALUES:
            graph4(run_dir, values, 'Signal', RES, Config.DPI, Config.ROUNDS)
            graph4(run_dir, values, 'Failure', RES, Config.DPI, Config.ROUNDS)

    if Config.ROUNDS > 1:
        for number in range(22, 47):
            for values in range(len(Params.VAR_2_VALUES)):
                graph3(run_dir, number, RES, Config.DPI, values, 'Blues_d')
        for values in Params.VAR_1_VALUES:
            graph2(run_dir, values, 'Errors', RES,
                   Config.ROUNDS, Config.DPI)
            graph2(run_dir, values, 'Reaction', RES,
                   Config.ROUNDS, Config.DPI)
            graph2(run_dir, values, 'Error Rate', RES,
                   Config.ROUNDS, Config.DPI)
            graph2(run_dir, values, 'Feedback Omission', RES,
                   Config.ROUNDS, Config.DPI)
            graph2(run_dir, values, 'Feedback Commission', RES,
                   Config.ROUNDS, Config.DPI)
            graph2(run_dir, values, 'Near Miss Detection', RES,
                   Config.ROUNDS, Config.DPI)


def write_csv(csv_path, RES):
    from main import Config, Params

    with open(csv_path, 'w') as OUTFILE:
        OUTFILE.write('sep=,\n')
        first_row = list()
        OUTFILE.write('Round')
        OUTFILE.write(',')
        for title in range(Params.NO_ATTRIBUTES):
            text = Params.COLUMNS[title][1]
            text = str(text)
            first_row.append(text)
            OUTFILE.write(text)
            if title != Params.NO_ATTRIBUTES - 1:
                OUTFILE.write(',')
        OUTFILE.write('\n')
        for round_no in range(Config.ROUNDS):
            for row in range(
                    len(Params.VAR_1_VALUES) * len(Params.VAR_2_VALUES)):
                OUTFILE.write(str(round_no + 1))
                OUTFILE.write(',')
                for column in range(Params.NO_ATTRIBUTES):
                    # print(row)
                    OUTFILE.write(str(RES[row, round_no, column]))
                    if column != Params.NO_ATTRIBUTES - 1:
                        OUTFILE.write(',')
                OUTFILE.write('\n')
