# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:52:51 2017
Python 3.x
@author: Helge Klapper
"""
import os
import numpy as np


def init_path_field(x, y, mean, range_x, range_y):
    """
    Create field that on average has mean latent error prob.
    For each row, sample a mean from a uniform distribution (clipped to [0.01, 0.99]).
    For each cell in the row, sample probability using a uniform distribution with that row's mean and var_y, also clipped.
    """
    # For rows, sample means uniformly around [mean-var_x, mean+var_x]
    low_row = max(0.01, mean - range_x)
    high_row = min(0.99, mean + range_x)
    row_means = np.random.uniform(low=low_row, high=high_row, size=y)
    # print("Row means:", row_means)

    field = np.zeros((y, x))
    for i in range(y):
        # For each cell, sample uniformly around [row_mean-var_y, row_mean+var_y]
        low_cell = max(0.01, row_means[i] - range_y)
        high_cell = min(0.99, row_means[i] + range_y)
        cell_probs = np.random.uniform(low=low_cell, high=high_cell, size=x)
        field[i, :] = cell_probs

    return field


def place_errors(x, y, field, pr_e_field, reset, failure):
    """
    Place all new potential errors
    """
    if reset == 1 and failure == 1:
        return np.zeros((y, x))

    magic_field = np.random.random((y, x))
    new_field = np.where(magic_field < pr_e_field, 0, 1)
    return 1 - np.multiply((1 - field), new_field)


def pathogen_trigger(x, y, error_field, prob_active):
    """
    This function determines whether pathogens will be triggered.
    Depending on activation probability, pathogens will be "activated"
    """
    random_field = np.random.rand(y, x)
    relevant_field = np.multiply(random_field, error_field)
    return np.round(np.greater(relevant_field, 1 - prob_active))


def init_agents(x, y, n, central):
    """
    Iniatilizes which coordinates belong to which.  
    If centralized, all agents can be everywhere.
    If decentralized, agents are assigned to specific coordinates.
    """
    agent_positions = np.full((y, x), -1)
    if central == 1:
        # If central, all agents everywhere
        return agent_positions
    elif central == 0:
        # Distribute agents among rows first, then columns
        agents_per_row = n // y
        extra_agents = n % y
        agent_id = 0
        for row in range(y):
            # Distribute extra agents to the first 'extra_agents' rows
            num_agents_this_row = agents_per_row + (1 if row < extra_agents else 0)
            cols_per_agent = x // num_agents_this_row
            extra_cols = x % num_agents_this_row
            col_start = 0
            for i in range(num_agents_this_row):
                # Distribute extra columns to the first 'extra_cols' agents in this row
                num_cols = cols_per_agent + (1 if i < extra_cols else 0)
                col_end = col_start + num_cols
                agent_positions[row, col_start:col_end] = agent_id
                agent_id += 1
                col_start = col_end

    return agent_positions


def place_agents(x, y, n, agent_pos, belief_mat, tau):
    """
    Place agents on field, either autonomously or by a manager.
    Use softmax to determine where agents are placed.
    """
    # if belief_mat is a 1D array, copy values it to (y, x)
    if belief_mat.ndim == 1:
        belief_mat = np.tile(belief_mat[:, np.newaxis], (1, x))
    agent_locations = np.zeros((y, x), dtype=int)
    available_pos = np.copy(agent_pos)
    # print("Available positions:\n", available_pos)

    for agent in range(n):
        # make a list of all coordinates in agent_post that either are -1 or the agent's number
        agent_coordinates = np.where((available_pos == agent) | (available_pos == -1))
        coords = list(zip(agent_coordinates[0], agent_coordinates[1]))
        # print(f"Agent {agent} coordinates: {coords}")
        # find the probabilities in belief_mat for these coordinates
        belief_probs = np.array([belief_mat[row, col] for row, col in coords])
        # print(f"Belief probabilities for agent {agent}: {belief_probs}")
        # Use softmax with parameter tau to determine probabilities
        belief_probs = np.exp(belief_probs / tau)
        belief_probs /= np.sum(belief_probs)
        # print(f"Softmax probabilities for agent {agent}: {belief_probs}")
        # Choose a coordinate based on the probabilities
        idx = np.random.choice(len(coords), p=belief_probs)
        row, col = coords[idx]
        #  print(f"Agent {agent} placed at: ({row}, {col})")
        # Mark the coordinate as occupied by the agent, indicated by -2
        agent_locations[row, col] = 1
        # Remove the coordinate from available positions
        available_pos[row, col] = -2

    return agent_locations


def reporting(x, y, agent_locations, rep_error, error_field):
    """
    Function describing whether individuals report an error
    """
    magic_field = np.random.random((y, x))
    # Fields where agents are located and have an error (error_field==1) will be reported with prob 1 - rep_error
    report1 = np.where((agent_locations == 1) & (error_field == 1) & (magic_field < (1 - rep_error)), 1, 0)
    # Fields where agents are located and do not have an error (error_field==0) will be reported with rep_error
    report2 = np.where((agent_locations == 1) & (error_field == 0) & (magic_field < rep_error), 1, 0)
    reported_fields = report1 + report2
    return reported_fields


def field_test(x, y, field):
    """
    Test whether failure occurs
    """
    ffield = np.zeros((y, x))
    ffield[0, :] = field[0, :]
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
        return failure

    failure = 1
    return failure


def org_investigate(x, y, interpret, org_check, org_listening, divisions,
                    int_w):
    """
    Function describing how organization aggregates reports from agents.
    Assuming that organization actually investigaes.
    0 for listening to signal
    1 for listening to agents
    """
    org_int = np.zeros((y, x))
    org_ag_int = np.zeros((y, x))

    random_order = np.arange((x * y))
    np.random.shuffle(random_order)
    int_w_f = int_w.flatten('F')
    reported = np.sum(interpret)
    # print('Int map\n', interpret)
    pos_reports_y = np.where(interpret)[0]
    pos_reports_x = np.where(interpret)[1]
    # print('Position reported', pos_reports_y)
    # print('Position reported', pos_reports_x)
    shuffled_number = np.arange(reported)
    shuffled_number.astype(int)
    # print('Shuffled numbers', shuffled_number)
    np.random.shuffle(shuffled_number)
    # print('Shuffled numbers', shuffled_number)

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
                org_ag_int[y_c, x_c] = 1
                org_check -= 1
            else:
                org_check = 0
            # print('Checks available agents', org_check)
            while org_check > 0 and iteration < reported:
                shuffled = int(shuffled_number[iteration])
                iteration += 1
                # print('Position', shuffled, 'Iteration', iteration)
                x_c = pos_reports_x[shuffled]
                y_c = pos_reports_y[shuffled]
                # print('Position', y_c, x_c)
                if org_int[y_c, x_c] == 0:
                    # Agent interpretation
                    # If there is a reporter error, investigate it
                    org_int[y_c, x_c] = 1
                    org_ag_int[y_c, x_c] = 1
                    org_check -= 1
                    # print('Checks left', org_check)
            # while org_check > 0 and iteration < (x*y):
            #     number = random_order[iteration]
            #     iteration += 1
            #     # print('Position', number, 'Iteration', iteration)
            #     x_c, y_c = np.divmod(number, y)
            #     # print('Divmod y and x', y_c, x_c)
            #     if org_int[y_c, x_c] == 0:
            #         # Agent interpretation
            #         # If there is a reporter error, investigate it
            #         org_int[y_c, x_c] = 1
            #         org_check -= 1
            #         # print('Checks left', checks_left)

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
                    if (interpret[y_c, x_c] == 1 and org_int[y_c, x_c] == 0):
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
    # print('Org Int\n', org_int)
    # print('Org Agents Int\n', org_ag_int)
    return org_int, org_ag_int


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


def update_beliefs(report, belief_mat, round_no):
    """
    Updates beliefs based on agent reports.
    If report==1, belief decreases; if report==0, belief increases.
    The effect diminishes as round_no increases.
    """
    print("Report matrix:\n", report, "Round number:", round_no)
    updating = (2 * report - 1) / (round_no + 1)
    print("Updating belief matrix with:\n", updating)
    belief_mat = belief_mat + updating
    # Optionally clip values:
    # belief_mat = np.clip(belief_mat, 0, 1)
    return belief_mat


# Start loop here
def simulation(args):
    # The processes inherits the same state for the random generator
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    # DEFINING MODEL INTERNAL OBJECTS
    # Whether group decision was correct
    near_det = np.zeros((args.E, args.ROUNDS))
    near_det_roll = np.zeros((args.E, args.ROUNDS))
    failure_ave = np.zeros((args.E, args.ROUNDS))
    failure_roll = np.zeros((args.E, args.ROUNDS))
    failure_dummy = np.zeros((args.E, args.ROUNDS))
    failure = np.zeros((args.E, args.ROUNDS))
    errors = np.zeros((args.E, args.ROUNDS))
    pct_reported = np.zeros((args.E, args.ROUNDS))
    pct_inv_agents = np.zeros((args.E, args.ROUNDS))
    pct_inv_cap = np.zeros((args.E, args.ROUNDS))
    pct_repaired = np.zeros((args.E, args.ROUNDS))
    omission = np.zeros((args.E, args.ROUNDS))
    commission = np.zeros((args.E, args.ROUNDS))
    ind_error = np.zeros((args.E, args.ROUNDS))
    agents_correct = np.zeros((args.E, args.ROUNDS))
    agents_percentage = np.zeros((args.E, args.ROUNDS))
    info_error = np.zeros((args.E, args.ROUNDS))


    for e in range(args.E):
        """Initializing each run"""
        # Check if next line is needed
        prob_e = args.PROB_E
        error_post = 0
        failure_d = 0
        failure_a = 0
        prob_e_field = init_path_field(args.X, args.Y,
                                       prob_e, args.PROB_E_SD_Y, args.PROB_E_SD_X)
        # Multiply the right side of the field by 2 and divide the left side by 2
        # mid = args.X // 2
        # prob_e_field[:, :mid] = prob_e_field[:, :mid] * 0.5
        agent_pos = init_agents(args.X, args.Y, args.N, args.CENTRAL)
        # print("Agent positions:\n", agent_pos)
        error_field = np.zeros((args.Y, args.X))
        report_freq = np.zeros((args.Y, args.X))
        belief_mat_cen = np.zeros((args.Y))
        belief_mat = np.zeros((args.Y, args.X))
        repair_wait = np.full((args.Y, args.X), -1, dtype=int)

        for round_no in range(args.ROUNDS):
            # print("Round number:", round_no)
            # First, update pathogen/causes
            error_field = place_errors(args.X, args.Y, error_field, prob_e_field,
                                       args.RESET, error_post)
            # Placing agents on the board
            if args.CENTRAL == 0:
                agent_locations = place_agents(args.X, args.Y, args.N, agent_pos, belief_mat, args.TAU)
            else:            
                agent_locations = place_agents(args.X, args.Y, args.N, agent_pos, belief_mat_cen, args.TAU)

            # Agents intepretation
            reports = reporting(args.X, args.Y, agent_locations, args.REP_ERROR, error_field)

            # Organization's observation of situation
            # org_report = organization_report(observe)
            # ag_report = agents_report(interpret, args.N, args.DEC_STRU)
            ag_non_report = agent_locations - reports
            omit = np.sum(np.multiply(ag_non_report, error_field))
            commit = np.sum(np.multiply(reports, 1 - error_field))
            no_fields_report = np.sum(reports)
            corr_reports = no_fields_report - commit

            # which field(s) does the org check?
            if args.CENTRAL == 0:
                repair_field = reports
                no_fields_inv = np.sum(repair_field)
            else:
                repair_wait = np.where((reports == 1) & (repair_wait == -1), args.CEN_DELAY, repair_wait)
                no_fields_inv = np.sum(np.where(repair_wait == args.CEN_DELAY, 1, 0))
                # print('fields investigates by agents', no_fields_inv)
                repair_field = np.where(repair_wait == 0, 1, 0)
                repair_wait = repair_wait - 1
                repair_wait = np.where(repair_wait < -1, -1, repair_wait)
            error_field = np.where(repair_field == 1, 0, error_field)
            no_fields_repaired = np.sum(repair_field)
            report_freq = report_freq + reports
            # print('Fields repaired', no_fields_repaired)
            belief_mat = report_freq / (round_no + 1)
            belief_mat_cen = np.mean(belief_mat, axis=1)

            error_post = field_test(args.X, args.Y, error_field)
            failure[e, round_no] = error_post
            # Agents update their beliefs about errors
            
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
            errors[e, round_no] = np.mean(np.floor(error_field))

            
            agents_correct[e, round_no] = corr_reports / args.N
            if no_fields_report > 0:
                pct_inv_agents[e, round_no] = no_fields_inv / no_fields_report
                pct_inv_cap[e, round_no] = no_fields_inv / args.ORG_CAP
                agents_percentage[e, round_no] = corr_reports / no_fields_report

            else:
                pct_inv_agents[e, round_no] = np.NaN
                agents_percentage[e, round_no] = np.NaN
                pct_inv_cap[e, round_no] = np.NaN

            if no_fields_inv > 0:
                # pct_repaired[e, round_no] = (no_fields_repaired)
                pct_repaired[e, round_no] = (no_fields_repaired /
                                             no_fields_inv)
            else:
                pct_repaired[e, round_no] = np.NaN
            
            pct_reported[e, round_no] = no_fields_report / args.N
            omission[e, round_no] = omit / args.N
            commission[e, round_no] = commit / args.N
            ind_error[e, round_no] = (omit + commit) / args.N
            if args.CENTRAL == 0:
                info_error[e, round_no] = 1 - np.corrcoef(
                    belief_mat.flatten(), prob_e_field.flatten())[0, 1]
            else:
                info_error[e, round_no] = 1 - np.corrcoef(
                    np.tile(belief_mat_cen, args.X).flatten(), prob_e_field.flatten())[0, 1]

        # print("Error field at end of run", e, ":\n", np.round(prob_e_field, 2))
        # print("Belief matrix at end of run", e, ":\n", np.round(belief_mat, 2))

    # Result Matrix
    r_a = np.zeros((1, args.ROUNDS, len(args.COLUMNS)))

    # The same for all rounds
    r_a[0, :, 0] = args.E
    r_a[0, :, 1] = args.N
    r_a[0, :, 2] = args.X
    r_a[0, :, 3] = args.Y
    r_a[0, :, 4] = prob_e
    r_a[0, :, 5] = args.PROB_E_SD_Y
    r_a[0, :, 6] = args.PROB_E_SD_X
    r_a[0, :, 7] = args.PROB_A
    r_a[0, :, 8] = args.START_E
    r_a[0, :, 9] = args.TAU
    r_a[0, :, 10] = args.RESET
    r_a[0, :, 11] = args.ORG_CAP
    r_a[0, :, 12] = args.MIDDLE
    r_a[0, :, 13] = args.CENTRAL
    r_a[0, :, 14] = args.REP_ERROR
    r_a[0, :, 15] = args.CEN_DELAY

    # Fill the whole column in 1 go
    r_a[0, :, 16] = np.sum(errors, axis=0) / args.E
    r_a[0, :, 17] = np.sum(pct_reported, axis=0) / args.E
    r_a[0, :, 18] = np.nanmean(pct_inv_agents, axis=0)
    r_a[0, :, 19] = np.nanmean(pct_inv_cap, axis=0)
    r_a[0, :, 20] = np.nanmean(pct_repaired, axis=0)
    r_a[0, :, 21] = np.sum(omission, axis=0) / args.E
    r_a[0, :, 22] = np.sum(commission, axis=0) / args.E
    r_a[0, :, 23] = np.sum(ind_error, axis=0) / args.E
    r_a[0, :, 24] = np.nanmean(agents_correct, axis=0)
    r_a[0, :, 25] = np.nanmean(agents_percentage, axis=0)
    r_a[0, :, 26] = np.sum(failure, axis=0) / args.E
    r_a[0, :, 27] = np.sum(failure_roll, axis=0) / args.E
    r_a[0, :, 28] = np.sum(failure_ave, axis=0) / args.E
    r_a[0, :, 29] = np.sum(failure_dummy, axis=0) / args.E
    r_a[0, :, 30] = np.sum(info_error, axis=0) / args.E
    return r_a
