# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:52:51 2017
Python 3.x
@author: Helge Klapper
"""
import os
import numpy as np


# TODO: reverse signs of weight on agents
def init_path_field(x, y, pr_e_mean, pre_e_sd):
    """
    Create field that on average has mean latent error prob.
    """
    if pre_e_sd > 0:
        alpha = ((1 - pr_e_mean) / pre_e_sd ** 2 - (1 / pr_e_mean)) * pr_e_mean ** 2
        beta = alpha * ((1 / pr_e_mean) - 1)
        return np.random.beta(alpha, beta, (y, x))
    else:
        return np.ones((y, x)) * pr_e_mean


def update_path_field(x, y, pr_e_field, field, coupling):
    """
    Update field to take into account added because of infected neighbors
    """
    new_pr_field = np.ones([y, x])
    # print('Error prob field\n', pr_e_field)
    # print('Latent error field\n', field)
    if coupling == 0:
        return pr_e_field
    for col in range(x):
        for row in range(y):
            # Check all possible neighbors
            # print('Position', col, row)
            if col == 0:
                top = col
                bottom = col + 1
            elif col == x:
                top = col - 1
                bottom = col
            else:
                top = col - 1
                bottom = col + 1
            if row == 0:
                left = row
                right = row + 1
            elif row == y:
                left = row - 1
                right = row
            else:
                left = row - 1
                right = row + 1
            # print(bottom, top, left, right)
            # Sum all neighbors that have an (latent) error
            sum_errors = np.sum(field[left:right+1, top:bottom+1]) + 1
            new_pr_field[row, col] = pr_e_field[row, col] * sum_errors
            # print(sum_errors)
    return new_pr_field


def place_errors(x, y, field, pr_e_field, reset, failure):
    """
    Place all new potential errors
    """
    if reset == 1 and failure == 1:
        return np.zeros((y, x))

    magic_field = np.random.random((y, x))
    new_field = np.where(magic_field < pr_e_field, 0, 1)
    return 1 - np.multiply((1 - field), new_field)


def pathogen_signals(x, y, error_field, noise, normal):
    """
    This function shows what agents will see. If there is an error,
    agents will receive a signal from a triangular distribution (0,1,1).
    If there is no error, will receive signal from a
    different triangular distribution (0,0,1)
    """
    if normal == 0:
        field_okay = np.random.triangular(0.5 - noise, 1, 1, (y, x))
        field_error = np.random.triangular(0, 0, 0.5 + noise, (y, x))
        return np.where(error_field == 1, field_error, field_okay)
    elif normal == 1:
        # not working yet
        noise_field = np.full((y, x), noise)
        return np.random.normal(error_field, noise_field, (y, x))


def pathogen_trigger(x, y, error_field, prob_active):
    """
    This function determines whether pathogens will be triggered.
    Depending on activation probability, pathogens will be "activated"
    """
    random_field = np.random.rand(y, x)
    relevant_field = np.multiply(random_field, error_field)
    return np.round(np.greater(relevant_field, 1 - prob_active))


def init_agents(n, x, tend, tend_sd):
    """
    Iniatilizes agents with skills and other variables
    """
    alpha = ((1 - tend) / tend_sd ** 2 - (1 / tend)) * tend ** 2
    beta = alpha * ((1 / tend) - 1)
    agent_vector = np.random.beta(alpha, beta, [n])
    agent_columns = np.arange(n) % x
    return agent_vector, agent_columns


def place_agents(x, y, n, agent_columns):
    """
    Place agents with on field
    """
    agent_locations = np.full((y, x), -1)
    agent_locations2 = np.zeros((y, x))
    agent_locations_nos = []

    for entry in range(n):
        coord_x = agent_columns[entry]
        while True:
            coord_y = np.random.choice(y)
            if agent_locations[coord_y, coord_x] == -1:
                # First one gives the number of the worker
                agent_locations[coord_y, coord_x] = entry
                # NOS is list of coordinate/agent tuple
                agent_locations_nos.append((coord_y, coord_x, entry))
                # Second is just binary occupied/free
                agent_locations2[coord_y, coord_x] = 1
                break

    return agent_locations, agent_locations2, agent_locations_nos


def field_test(x, y, field, full_field=1):
    """
    Test whether failure occurs
    """
    ffield = np.zeros((y, x))
    ffield[0, :] = field[0, :]
    ff2 = np.zeros((y, x))
    field = np.floor(field)

    current_entrants = [(0, i) for i in np.where(field[0, :] == 1)[0]]
    while current_entrants:
        row, col = current_entrants.pop(0)
        # Down
        if row < y - 1:
            if field[row + 1, col] == 1 and ffield[row + 1, col] == 0:
                ffield[row + 1, col] = 1
                current_entrants.append((row + 1, col))
        # Left
        if col > 0:
            if field[row, col - 1] == 1 and ffield[row, col - 1] == 0:
                ffield[row, col - 1] = 1
                current_entrants.append((row, col - 1))
        # Right
        if col < x - 1:
            if field[row, col + 1] == 1 and ffield[row, col + 1] == 0:
                ffield[row, col + 1] = 1
                current_entrants.append((row, col + 1))

    # Last row
    failure = 0
    if np.sum(ffield[y - 1, :]) == 0:
        if np.sum(ffield[y - 2, :]) > 0:
            near_miss = 1
            return near_miss, failure, ff2
        else:
            near_miss = 0
            return near_miss, failure, ff2

    failure = 1
    near_miss = 0
    if full_field != 1:
        return near_miss, failure, ff2

    current_entrants = list()
    for col in range(x):
        ff2[y - 1, col] = ffield[y - 1, col]
        if ffield[y - 1, col] == 1:
            current_entrants.append((y - 1, col))

    while current_entrants:
        row, col = current_entrants.pop(0)
        # Up
        if row > 0:
            if ffield[row - 1, col] == 1 and ff2[row - 1, col] == 0:
                ff2[row - 1, col] = 1
                current_entrants.append((row - 1, col))
        # Left
        if col > 0:
            if ffield[row, col - 1] == 1 and ff2[row, col - 1] == 0:
                ff2[row, col - 1] = 1
                current_entrants.append((row, col - 1))
        # Right
        if col < x - 1:
            if ffield[row, col + 1] == 1 and ff2[row, col + 1] == 0:
                ff2[row, col + 1] = 1
                current_entrants.append((row, col + 1))

    return near_miss, failure, ff2


def field_test_ind(x, y, field, full_field=1):
    """
    Test whether failure occurs
    """
    ffield = np.zeros((y, x))
    ffield[0, :] = field[0, :]
    ff2 = np.zeros((y, x))
    field = np.floor(field)
    failure = 0
    near_miss = 0

    systems = np.sum(field, axis=0)
    # print('systems', systems)
    failure = np.any(systems == y)
    # print('failure', failure)

    if failure == 0 and np.any(systems == y-1) == 1:
        near_miss = 1

    for row in range(x):
        if systems[row] == y:
            ff2[:, row] = np.ones((y))

    return near_miss, failure, ff2


def which_test(linear, x, y, field, full_field=1):
    if linear == 0:
        near_miss, failure, ff2 = field_test(x, y, field, full_field)
    elif linear == 1:
        near_miss, failure, ff2 = field_test_ind(x, y, field, full_field)
    return near_miss, failure, ff2


def raw_interpret(x, y, signal_field, location_idx_sets, agent_tend):
    """
    Function describing whether individuals see signal as problematic
    """
    signal_int = np.zeros((y, x))
    signal_weight = np.zeros((y, x))
    for row, col, idx in location_idx_sets:
        if signal_field[row, col] > 0:
            # Agent interpretation : if there is a clear error,
            # which is larger than the threshold, report it
            if signal_field[row, col] < agent_tend[idx]:
                test = signal_field[row, col]
                tend = agent_tend[idx]
                signal_int[row, col] = 1
                signal_weight[row, col] = tend - test
    return signal_int, signal_weight


def agents_report(interpret_field, n, dec_structure):
    """
    Describes conditional on the decision structure whether
    agents report something
    0: As long as one person reports
    1: Majority of agents
    2: Consensus
    If between 0 and 1, describes the necessary percentage
    """
    # How many agents report
    reporters = np.sum(interpret_field)
    # One person necessary
    if dec_structure == 0:
        if reporters > 0:
            return 1
    # Majority
    elif dec_structure == 1:
        if reporters >= np.round(n / 2, 0):
            return 1
    # Consensus
    elif dec_structure == 2:
        if reporters == n:
            return 1
    elif dec_structure > 0 and dec_structure < 1:
        if reporters >= dec_structure * n:
            return 1
    return 0


def organization_report(observe):
    """
    Determines based on number of errors whether overall signal indicates
    failure
    """
    return observe > np.random.rand()


def org_listen(org_weight):
    """
    Conditional on weight whether to listen to agents or org.
    False for listening to signal
    True for listening to agents
    """
    return org_weight > np.random.random()


def org_decision(org_listening, observe, agent_report):
    """
    Function describing whether organization is going to investigate.
    0 NOT to investigate, 1 to investigate
    """
    # Check whether signals differ
    if org_listening:
        # Listening to agents
        return agent_report
    else:
        # Listening to observation
        return observe


def org_investigate(x, y, interpret, org_check, org_listening, divisions,
                    int_w):
    """
    Function describing how organization aggregates reports from agents.
    Assuming that organization actually investigaes.
    0 for listening to signal
    1 for listening to agents
    """
    org_int = np.zeros((y, x))

    random_order = np.arange((x * y))
    np.random.shuffle(random_order)
    int_w_f = int_w.flatten('F')

    if org_listening:
        if divisions == 1:
            iteration = 0
            # if organization decides to go with agent interpretation
            # int_w is the matrix with difference between threshold and signal
            # First check is to determine the agent who has largest difference
            # between own threshold and signal, i.e. the most concerned
            max_index = np.argmax(int_w_f)
            # print('Max arg', max_index)
            x_c, y_c = np.divmod(max_index, y)
            # print('Divmod y and x', y_c, x_c)
            if interpret[y_c, x_c] == 1:
                # print('First check at', y_c, x_c)
                org_int[y_c, x_c] = 1
                org_check -= 1
            else:
                org_check = 0
            # print('Checks available agents', org_check)
            while org_check > 0 and iteration < (x*y):
                number = random_order[iteration]
                iteration += 1
                # print('Position', number, 'Iteration', iteration)
                x_c, y_c = np.divmod(number, y)
                # print('Divmod y and x', y_c, x_c)
                if (interpret[y_c, x_c] == 1 and
                    org_int[y_c, x_c] == 0):
                    # Agent interpretation
                    # If there is a reporter error, investigate it
                    org_int[y_c, x_c] = 1
                    org_check -= 1
                    # print('Checks left', checks_left)
        elif divisions > 1:
            # print('Middle managers here')
            len_division = int(np.floor(np.divide(int(x*y), divisions)))
            for div in range(divisions):
                iteration = 0
                # print('Div. number:', div)
                div_start = div * len_division
                checks_left = int(np.round(np.divide(org_check, divisions)))
                rel_slice = int_w_f[div_start: (div_start + len_division)]
                # relevant slice of
                max_index = np.argmax(rel_slice)
                max_index = max_index + div_start
                # print('Index for coords', max_index)
                x_c, y_c = np.divmod(max_index, y)
                # print('Divmod y and x', y_c, x_c)
                if interpret[y_c, x_c] == 1:
                    # print('First check at', y_c, x_c)
                    org_int[y_c, x_c] = 1
                    checks_left -= 1
                else:
                    # if not even one positive check, go to next divisions
                    continue
                # print('Checks left', checks_left)
                locations = np.arange(div_start, div_start + len_division)
                # print('locations', locations)
                np.random.shuffle(locations)
                # print('Random locations', locations)
                while checks_left > 0 and iteration < len_division:
                    number = locations[iteration]
                    iteration += 1
                    # print('Position', number, 'Iteration', iteration)
                    x_c, y_c = np.divmod(number, y)
                    # print('Y coordinate', y_c)
                    # print('X coordinate', x_c)
                    if (interpret[y_c, x_c] == 1 and
                        org_int[y_c, x_c] == 0):
                        # Agent interpretation
                        # If there is a reporter error, investigate it
                        org_int[y_c, x_c] = 1
                        checks_left -= 1

    else:
        # if org goes with observation
        # which fields to look at
        for number in random_order[:org_check]:
            org_int[np.divmod(number, x)] = 1

    # All occuring random checks
    return org_int


def repair(x, y, error_field, interpretation, org_detect):
    """
    Repairs the found pathogens, depending on org. detection
    capability. For each cell, random draw whether can be detected
    this round.
    """
    # Determine all cells that can be detected
    magic_field = np.random.random((y, x))
    magic_field2 = np.where(magic_field < org_detect, 1, 0)
    detectable_field = np.multiply(magic_field2, error_field)
    # Check whether detectable cells were investigated.
    # Those are then repaired
    return np.multiply(detectable_field, interpretation)


def update_cause_field(cause_field, repair_field):
    non_repair = 1 - repair_field
    return np.multiply(cause_field, non_repair)


def feedback(location_idx_sets, tends, repair_field, interpret,
             failure_field, d_up, d_down, org_int):
    """
    How individuals adapt their threshold depending on outcome and
    own decision
    """
    fb_failure = 0
    fb_comm = 0
    fb_omit = 0
    for row, col, agent_no in location_idx_sets:
        # Did failure happen in my column?
        failure_column = np.sum(failure_field[:, col])
        reported = interpret[row, col]

        if failure_column == 0:
            # Boss check
            if not org_int[row, col]:
                continue

            found = repair_field[row, col]
            if reported == 0:
                if found == 1:
                    tends[agent_no] += (1 - tends[agent_no]) * d_up
                    fb_omit += 1
            else:
                if found == 0:
                    # Org investigates, but finds no problem
                    tends[agent_no] -= tends[agent_no] * d_down
                    fb_comm += 1
        else:
            if reported == 0 and failure_field[row, col] == 1:
                # punishment because impending failure not foreseen
                tends[agent_no] += (1 - tends[agent_no]) * d_up
                fb_failure += 1
    return tends, fb_failure, fb_comm, fb_omit


def org_feedback(no_fields_inv, no_fields_rep, org_dec, delta_org, org_weight,
                 org_listening, failure, org_thresh, org_check,
                 org_check_change):
    """
    Update organizational weighting and threshold to react to problem
    """
    if org_dec == 1:
        if no_fields_inv == 0:
            correct_ratio = 0
        else:
            correct_ratio = no_fields_rep / no_fields_inv

        # if listenening to agents
        if org_listening == 1:
            to_agent = int(correct_ratio > org_thresh)
            # capacity change, if wrong reduce, if right increase
            if org_check_change == 1 and no_fields_rep > org_check:
                org_check = org_check - 1 + 2 * to_agent
        # if listening to observation
        else:
            to_agent = int(correct_ratio <= org_thresh)
            if org_check_change == 1:
                org_check = org_check + 1 - 2 * to_agent
    else:
        # if it was not a failure, thumps up!
        if failure == 1:
            # WRONG
            to_agent = int(not org_listening)
            if org_check_change == 1:
                org_check += 1
        else:
            # RIGHT
            to_agent = int(org_listening)
            # if not reported, why reduce?
#            if org_check_change == 1:
#                org_check -= 1
    org_check = np.maximum(1, org_check)
    if to_agent == 1:
        # if in direction of agents
        return org_weight + (1 - org_weight) * delta_org, 1, org_check
    else:
        # if in direction of observation
        return org_weight - org_weight * delta_org, 0, org_check


def org_fb(no_fields_inv, no_fields_rep, delta_org, org_weight, org_listening,
           failure, org_thresh, org_check, org_check_change):
    """
    Update organizational weighting and threshold to react to problem
    """
    if no_fields_inv == 0:
        correct_ratio = 0
    else:
        correct_ratio = no_fields_rep / no_fields_inv

    # if listenening to agents
    if org_listening == 1:
        to_agent = int(correct_ratio > org_thresh)
        # capacity change, if wrong reduce, if right increase
        if org_check_change == 1 and no_fields_rep > org_check:
            org_check = org_check - 1 + 2 * to_agent
    # if listening to observation
    else:
        to_agent = -1

    org_check = np.maximum(1, org_check)
    # print('To Agent', to_agent)
    # print('Org Weight', org_weight)
    # print('Org Check', org_check)
    if to_agent == 1:
        # if in direction of agents
        # print(org_weight + (1 - org_weight) * delta_org, 1, org_check)
        return org_weight + (1 - org_weight) * delta_org, 1, org_check
    elif to_agent == 0:
        # if in direction of observation
        # print(org_weight - org_weight * delta_org, 0, org_check)
        return org_weight - org_weight * delta_org, 0, org_check
    else:
        # print(org_weight, -1, org_check)
        return org_weight, -1, org_check


def time_left(current_time, current_round, start, rounds):
    """Calculates time to finish"""
    passed = current_time - start
    time_per_round = passed / current_round
    return time_per_round * (rounds - current_round)


# Start loop here
def simulation(args):
    # The processes inherits the same state for the random generator
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    # DEFINING MODEL INTERNAL OBJECTS
    # Whether group decision was correct
    near_miss = np.zeros((args.E, args.ROUNDS))
    near_det = np.zeros((args.E, args.ROUNDS))
    near_det_ave = np.zeros((args.E, args.ROUNDS))
    near_det_roll = np.zeros((args.E, args.ROUNDS))
    failure_ave = np.zeros((args.E, args.ROUNDS))
    failure_roll = np.zeros((args.E, args.ROUNDS))
    failure_dummy = np.zeros((args.E, args.ROUNDS))
    failure = np.zeros((args.E, args.ROUNDS))
    errors = np.zeros((args.E, args.ROUNDS))
    pathogens_sum = np.zeros((args.E, args.ROUNDS))
    tend_ave = np.zeros((args.E, args.ROUNDS))
    tend_sd = np.zeros((args.E, args.ROUNDS))
    org_weight_mat = np.zeros((args.E, args.ROUNDS))
    listen_to_agent = np.zeros((args.E, args.ROUNDS))
    pct_reported = np.zeros((args.E, args.ROUNDS))
    pct_listened = np.zeros((args.E, args.ROUNDS))
    pct_repaired = np.zeros((args.E, args.ROUNDS))
    omission = np.zeros((args.E, args.ROUNDS))
    commission = np.zeros((args.E, args.ROUNDS))
    ind_error = np.zeros((args.E, args.ROUNDS))
    feedback_fail = np.zeros((args.E, args.ROUNDS))
    feedback_omit = np.zeros((args.E, args.ROUNDS))
    feedback_commit = np.zeros((args.E, args.ROUNDS))
    agents_correct = np.zeros((args.E, args.ROUNDS))
    agents_percentage = np.zeros((args.E, args.ROUNDS))
    org_correct = np.zeros((args.E, args.ROUNDS))
    org_check_mat = np.zeros((args.E, args.ROUNDS))

    for e in range(args.E):
        """Initializing each run"""
        if args.LINEAR == 1:
            prob_e = args.PROB_E
        else:
            prob_e = args.PROB_E
        org_weight = args.S_ORG_WEIGHT
        org_check = args.ORG_CHECK
        error_post = 0
        failure_d = 0
        failure_a = 0
        prob_e_field = init_path_field(args.X, args.Y,
                                       prob_e, args.PROB_E_SD)
        pathogens = np.random.choice(2, (args.Y, args.X), replace=True,
                                     p=[1 - args.START_E, args.START_E])
        agent_tends, agent_columns = init_agents(args.N, args.X, args.S_TEND,
                                                 args.TEND_SD)

        for round_no in range(args.ROUNDS):
            # First, update pathogen/causes
            updated_e_field = update_path_field(args.X, args.Y, prob_e_field,
                                                pathogens, args.COUPLING)
            pathogens = place_errors(args.X, args.Y, pathogens,
                                     updated_e_field, args.RESET, error_post)

            # Signals and Triggers are activated
            signal_field = pathogen_signals(args.X, args.Y, pathogens,
                                            args.NOISE, args.NORMAL)
            if args.PROB_A < 1:
                trigger_field = pathogen_trigger(args.X, args.Y, pathogens,
                                                 args.PROB_A)
            elif args.PROB_A == 1:
                trigger_field = pathogens

            # Placing agents on the board
            agent_locations, agent_locations2, agent_locations_nos = \
                place_agents(args.X, args.Y, args.N, agent_columns)

            # Agents intepretation
            interpret, int_weight = raw_interpret(args.X, args.Y, signal_field,
                                                  agent_locations_nos,
                                                  agent_tends)

            # Organization's observation of situation
            # observe = np.mean(pathogens)
            # org_report = organization_report(observe)
            ag_report = agents_report(interpret, args.N, args.DEC_STRU)
            ag_non_report = agent_locations2 - interpret
            omit = np.sum(np.multiply(ag_non_report, pathogens))
            commit = np.sum(np.multiply(interpret, 1 - pathogens))
            no_fields_report = np.sum(interpret)
            nm_t, error_t, t_field = which_test(args.LINEAR, args.X, args.Y,
                                                pathogens, 0)
            # nm_t, error_t, t_field = field_test(args.X, args.Y, pathogens, 0)

            # who does the organization listen to this round?
            org_listening = org_listen(org_weight)

            # which field(s) does the org check?
            org_int = org_investigate(args.X, args.Y, interpret, org_check,
                                      org_listening, args.MIDDLE, int_weight)
            no_fields_inv = np.sum(org_int)
            repair_field = repair(args.X, args.Y, pathogens, org_int,
                                  args.ORG_DETECT)
            prob_e_field = np.where(repair_field == 1,
                                    (prob_e_field * (1 - args.IMPROVE)),
                                    prob_e_field)
            no_fields_repaired = np.sum(repair_field)

            # Only update on change
            if np.any(repair_field):
                pathogens = update_cause_field(pathogens, repair_field)
                nm_t_post, error_t_post, t_field = \
                    which_test(args.LINEAR, args.X, args.Y, pathogens, 0)
                    # field_test(args.X, args.Y, pathogens, 0)
                trigger_field2 = pathogen_trigger(args.X, args.Y, pathogens,
                                                  args.PROB_A)
            else:
                error_t_post = error_t
                trigger_field2 = trigger_field

            nm_post, error_post, e_field_post = \
                which_test(args.LINEAR, args.X, args.Y, trigger_field2, 1)
            # field_test(args.X, args.Y, trigger_field2, 1)

            # Agents update their thresholds
            agent_tends, fb_f, fb_c, fb_o = feedback(agent_locations_nos,
                                                     agent_tends, repair_field,
                                                     interpret, e_field_post,
                                                     args.D_UP, args.D_DOWN,
                                                     org_int)

            org_weight, to_agent, org_check = org_fb(no_fields_inv,
                                                     no_fields_repaired,
                                                     args.D_ORG, org_weight,
                                                     org_listening,
                                                     error_post,
                                                     args.ORG_THRESH,
                                                     org_check,
                                                     args.ORG_CHECK_CHANGE)

            # Next line for testing
            near_miss[e, round_no] = nm_post
            if error_t == 1:
                near_det[e, round_no] = 1 - error_t_post
            else:
                near_det[e, round_no] = np.NaN
            near_det_ave[e, round_no] = np.nanmean(near_det[e, 0:round_no])
            failure[e, round_no] = error_post
            if error_post == 1:
                failure_d = 1
                failure_a += 1

            failure_dummy[e, round_no] = failure_d
            failure_ave[e, round_no] = failure_a / (round_no + 1)
            if round_no < 6:
                near_det_roll[e, round_no] = np.NaN
                failure_roll[e, round_no] = error_post
            else:
                r2 = round_no - 2
                sliced = near_det[e, r2:round_no]
                near_det_roll[e, round_no] = np.nanmean(sliced)
                failure_roll[e, round_no] = np.mean(failure[e, r2:round_no])
            pathogens_sum[e, round_no] = np.mean(pathogens)
            errors[e, round_no] = np.mean(np.floor(trigger_field))
            tend_ave[e, round_no] = np.mean(agent_tends)
            tend_sd[e, round_no] = np.std(agent_tends)
            listen_to_agent[e, round_no] = org_listening
            if org_listening == 1:
                agents_correct[e, round_no] = to_agent
                org_correct[e, round_no] = np.NaN
                if no_fields_report > 0:
                    liste = no_fields_inv / no_fields_report
                    pct_listened[e, round_no] = liste
                    no_corr = no_fields_report - commit
                    agents_percentage[e, round_no] = no_corr / no_fields_report
                else:
                    pct_listened[e, round_no] = 0
                    agents_percentage[e, round_no] = np.NaN
            else:
                org_correct[e, round_no] = np.NaN
                agents_correct[e, round_no] = np.NaN
                agents_percentage[e, round_no] = np.NaN
                pct_listened[e, round_no] = np.NaN
            org_weight_mat[e, round_no] = org_weight
            pct_reported[e, round_no] = no_fields_report / args.N

            if no_fields_repaired > 0:
                pct_repaired[e, round_no] = no_fields_repaired / no_fields_inv
            omission[e, round_no] = omit / args.N
            commission[e, round_no] = commit / args.N
            ind_error[e, round_no] = (omit + commit) / args.N
            feedback_fail[e, round_no] = fb_f / args.N
            feedback_omit[e, round_no] = fb_o / args.N
            feedback_commit[e, round_no] = fb_c / args.N
            org_check_mat[e, round_no] = org_check

    # Result Matrix
    r_a = np.zeros((1, args.ROUNDS, len(args.COLUMNS)))

    # The same for all rounds
    r_a[0, :, 0] = args.E
    r_a[0, :, 1] = args.N
    r_a[0, :, 2] = args.X
    r_a[0, :, 3] = args.Y
    r_a[0, :, 4] = prob_e
    r_a[0, :, 5] = args.PROB_E_SD
    r_a[0, :, 6] = args.PROB_A
    r_a[0, :, 7] = args.START_E
    r_a[0, :, 8] = args.S_TEND
    r_a[0, :, 9] = args.TEND_SD
    r_a[0, :, 10] = args.NOISE
    r_a[0, :, 11] = args.NORMAL
    r_a[0, :, 12] = args.S_ORG_WEIGHT
    r_a[0, :, 13] = args.ORG_THRESH
    r_a[0, :, 14] = args.D_UP
    r_a[0, :, 15] = args.D_DOWN
    r_a[0, :, 16] = args.D_ORG
    r_a[0, :, 17] = args.DEC_STRU
    r_a[0, :, 18] = args.ORG_CHECK
    r_a[0, :, 19] = args.ORG_CHECK_CHANGE
    r_a[0, :, 20] = args.RESET
    r_a[0, :, 21] = args.ORG_DETECT
    r_a[0, :, 22] = args.MIDDLE
    r_a[0, :, 23] = args.LINEAR
    r_a[0, :, 24] = args.COUPLING

    # Fill the whole column in 1 go
    r_a[0, :, 25] = np.sum(pathogens_sum, axis=0) / args.E
    r_a[0, :, 26] = np.sum(errors, axis=0) / args.E
    r_a[0, :, 27] = np.sum(tend_ave, axis=0) / args.E
    r_a[0, :, 28] = np.sum(tend_sd, axis=0) / args.E
    r_a[0, :, 29] = np.sum(pct_reported, axis=0) / args.E
    r_a[0, :, 30] = np.nanmean(pct_listened, axis=0)
    r_a[0, :, 31] = np.sum(pct_repaired, axis=0) / args.E
    r_a[0, :, 32] = np.sum(omission, axis=0) / args.E
    r_a[0, :, 33] = np.sum(commission, axis=0) / args.E
    r_a[0, :, 34] = np.sum(ind_error, axis=0) / args.E
    r_a[0, :, 35] = np.sum(feedback_fail, axis=0) / args.E
    r_a[0, :, 36] = np.sum(feedback_omit, axis=0) / args.E
    r_a[0, :, 37] = np.sum(feedback_commit, axis=0) / args.E
    r_a[0, :, 38] = np.sum(org_check_mat, axis=0) / args.E
    r_a[0, :, 39] = np.sum(org_weight_mat, axis=0) / args.E
    r_a[0, :, 40] = np.nanmean(org_correct, axis=00)
    r_a[0, :, 41] = np.nanmean(agents_correct, axis=00)
    r_a[0, :, 42] = np.nanmean(agents_percentage, axis=00)
    r_a[0, :, 43] = np.nanmean(near_miss, axis=00)
    r_a[0, :, 44] = np.nanmean(near_det, axis=00)
    r_a[0, :, 45] = np.nanmean(near_det_ave, axis=00)
    r_a[0, :, 46] = np.nanmean(near_det_roll, axis=00)
    r_a[0, :, 47] = np.sum(failure, axis=0) / args.E
    r_a[0, :, 48] = np.sum(failure_roll, axis=0) / args.E
    r_a[0, :, 49] = np.sum(failure_ave, axis=0) / args.E
    r_a[0, :, 50] = np.sum(failure_dummy, axis=0) / args.E


    return r_a
