import os
import pickle
import utils
import unused.model_bank as mb 
import numpy as np
import networkx as nx
from collections import defaultdict, deque
import pandas as pd
import copy
dt = 0.1

def get_files_in_path(path):
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files if os.path.isfile(os.path.join(path, file))]
    files.sort(key=os.path.getmtime)  # Sort files by last modified time
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            yield os.path.join(path, file)

def get_param_each_driver(model_name, parameter_path):#, read_file_name, write_file_name):
    """
    for each driver, find its leaders during the entire travel time, simulate its trajectory based on the calibrated model parameters saved in parameter_path
    if this driver has multiple parameters, pick the one that has lower obj func value (res.fun)
    if this driver has no parameters, use the default value [33.3, 2, 1.6, 0.73, 1.67]
    """
    model_name = model_name.lower()

    # select all the pkl files - pick lower obj func if multiple
    param_map = {}
    files_generator = get_files_in_path(parameter_path)

    # Initialize variables to keep track of the previous foll_id and its corresponding res with the lowest func
    prev_foll_id = None
    lowest_func_res = None

    # Read files one by one in a while loop
    while True:
        try:
            file_path = next(files_generator)
            with open(file_path, 'rb') as file:
                res = pickle.load(file)
                foll_id = file_path.split("_")[-2]

                # Check if the foll_id is different from the previous one or if the func is lower than the previous lowest_func_res
                if prev_foll_id is None or foll_id != prev_foll_id or (foll_id == prev_foll_id and res.fun < lowest_func_res.fun):
                    param_map[foll_id] = res.x
                    prev_foll_id = foll_id
                    lowest_func_res = res

        except StopIteration:
            # No more files in the directory
            break

    return param_map

def trace_driven_simulation(model_name, pair_id_file,  read_file_name, write_file_name, param_map=None):
    '''
    query parameters for each vehicle from calibrated result
    if no key -> set to default
    if param_map is None -> use default parameters to simulate
    '''
    append_items = ["timestamps", "lead_v"]
    prev_foll_id = None
    prev_cf_data = None

    with open(pair_id_file, 'r') as file:
        line_count = sum(1 for _ in file)

    cnt_sim_failed = 0
    cnt_default_param = 0

    # get default parameters
    if model_name == "cthrv":
        default_param = [0.1, 0.447, 1.7]
    elif model_name == "idm":
        default_param = [33.3, 2, 1.6, 0.73, 1.67]
    elif model_name == "ovrv":
        default_param = [3.14, 19.75, 22.2, 23.29]

    with open(write_file_name, "w") as write_file:
        with open(pair_id_file, 'r') as file:
            for idx in range(line_count):
                # if idx not in [440,441]:
                #     continue
                cf_data = utils.load_cf_data(read_file_name, pair_id_file, pair_id_idx=idx)
                foll_id = cf_data["foll_id"]

                if prev_foll_id is None:
                    prev_foll_id = foll_id
                    prev_cf_data = cf_data.copy()
                else:
                    try:
                        theta = param_map[str(prev_foll_id)]
                    except:
                        theta = default_param
                        cnt_default_param += 1
                    # theta = param_map.get(str(prev_foll_id), [33.3, 2, 1.6, 0.73, 1.67])
                    try:
                        prev_cf_data_sim = mb.solve_ivp_cf(model_name, prev_cf_data, theta)

                        # write follower trajectory to new file following NGSIM schema
                        for i, timestamp in enumerate(prev_cf_data_sim["timestamps"]):
                            output_row = [prev_cf_data_sim['foll_id'], timestamp*10, -1, prev_cf_data_sim["foll_p_sim"][i], 
                                        prev_cf_data_sim["foll_v_sim"][i],-1,-1,-1,-1,-1] # vehID, start pos, end pos, start time, start speed
                            line = " ".join(str(item) for item in output_row) + '\n'

                            write_file.write(line)

                    except Exception as e:
                        print(f"** Simulation error- foll_id: {prev_foll_id}, pair_id_idx: {idx}, message:{str(e)}")
                        cnt_sim_failed += 1
                        pass

                    prev_foll_id = foll_id
                    prev_cf_data = cf_data.copy()

    print(f"Trace-driven simulation completed. Total cf pairs:{line_count}, default param cnt: {cnt_default_param}, failed sim cnt: {cnt_sim_failed}")
    return




def intersection(intervals1, intervals2):
    """
    Finds the intersection of two lists of intervals with a buffer of 0.1.

    Args:
      intervals1: A list of intervals.
      intervals2: A list of intervals.

    Returns:
      A list of intervals that are the intersection of intervals1 and intervals2.
    """

    result = []
    for interval1 in intervals1:
        for interval2 in intervals2:
            # Check for overlaps with buffer
            if interval1[0] - 0.1 <= interval2[1] and interval1[1] + 0.1 >= interval2[0]:
                # Calculate the intersection without the buffer
                start = max(interval1[0], interval2[0])
                end = min(interval1[1], interval2[1])
                # Ensure the intersection is non-empty
                if start <= end:
                    result.append([start, end])

    return result


def union(intervals1, intervals2, buffer=0.11):
    """
    Finds the union of two lists of intervals with a buffer.

    Args:
      intervals1: A list of intervals.
      intervals2: A list of intervals.
      buffer: Buffer value for interval comparison.

    Returns:
      A list of intervals that represents the union of the two input lists.
    """

    # Check if intervals1 is empty
    if not intervals1:
        return intervals2
    # Check if intervals2 is empty
    if not intervals2:
        return intervals1

    # Sort the intervals by their start times.
    intervals1.sort(key=lambda x: x[0])
    intervals2.sort(key=lambda x: x[0])

    # Initialize a new list to hold the merged intervals.
    merged_intervals = []

    # Helper function to merge intervals with buffer
    def merge_with_buffer(start, end, new_start, new_end):
        if new_start - buffer <= end and start - buffer <= new_end:
            return [min(start, new_start), max(end, new_end)]
        else:
            return None

    # Iterate through intervals1 and intervals2
    i, j = 0, 0
    while i < len(intervals1) and j < len(intervals2):
        start1, end1 = intervals1[i]
        start2, end2 = intervals2[j]

        merged_interval = merge_with_buffer(start1, end1, start2, end2)
        if merged_interval:
            merged_intervals.append(merged_interval)
            i += 1
            j += 1
        elif end1 + buffer >= start2 and start1 - buffer <= end2:
            # If there is an overlap with buffer, merge the intervals
            merged_intervals.append([min(start1, start2), max(end1, end2)])
            i += 1
            j += 1
        elif end1 + buffer < start2:
            merged_intervals.append([start1, end1])
            i += 1
        elif end2 + buffer < start1:
            merged_intervals.append([start2, end2])
            j += 1

        # print(merged_intervals, i, j)

    # Add remaining intervals from intervals1 and intervals2
    while i < len(intervals1): # 
        merged_intervals.append(intervals1[i])
        i += 1
    while j < len(intervals2):
        merged_intervals.append(intervals2[j])
        j += 1

    # 
    merged_intervals2 = []
    if len(merged_intervals) == 0:
        return merged_intervals2
    start, end = merged_intervals[0]
    for i in range(1, len(merged_intervals)):
        next_start, next_end = merged_intervals[i]
        if next_start - end <= buffer:
            end = next_end
        else:
            merged_intervals2.append([start, end])
            start, end = next_start, next_end
    merged_intervals2.append([start, end])
    # return merged_intervals


    return merged_intervals2

def fixed_leader_simulation(model_name, pair_id_file, read_file_name, write_file_name, param_map=None):
    '''
    Similar to trace-driven simulation, but each leaders' profile is also simulated
    if a car has no leader, then their profile is read directly from the data
    This microsim should be less constrained, and less "overfitting" compared to trace-driven simulation
    '''
    # topological sort on the dependency graph
    # the graph is constructed such that each node is a vehicle ID
    # (directed) edges are connected from leader to follower
    # edge attribute is [start_time, end_time] of this car-following event
    # microsimulation should follow topological sort order
    
    # read all data in to a df
    column_names = ['Vehicle ID', 'Frame ID', 'Lane ID', 'LocalY',
                        'Mean Speed', 'Mean Acceleration', 'Vehicle length',
                        'Vehicle Class ID', 'Follower ID', 'Leader ID']
    df = pd.read_csv(read_file_name, sep='\t',
                 names=column_names)
    df = df.assign(timestamps=df['Frame ID'] * dt)
    df = df.drop(columns=['Frame ID'])

    # df.groupby("vehID")

    G = nx.DiGraph() # a dependency map
    # can_simulate = defaultdict(list)
    idx = 0
    with open(pair_id_file, 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()
            follower = int(columns[0])
            leader = int(columns[1])
            start_time = round(float(columns[2]),2)
            end_time = round(float(columns[3]),2)

            if G.has_edge(leader, follower):
                curr_edge_segments = G.get_edge_data(leader, follower)["segments"]
                curr_edge_segments.append([start_time, end_time])
                # print(leader, follower, curr_edge_segments)
                G.add_edge(leader, follower, segments=curr_edge_segments) #segments=union(curr_edge_segments, [start_time, end_time]))
            else:
                G.add_edge(leader, follower, segments=deque([[start_time, end_time]])) # from leader to follower
            # idx += 1
            # if idx > 20:
            #     break
    # Plot the graph
    # pos = nx.shell_layout(G)  # Positions for all nodes
    # nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=15)
    
    column_dict = {col: 1 for col in df.columns}
    column_dict["timestamps"] = 1/dt
    # filtered_dfs = []
    # all_reachable_edges = bfs_traverse_edges(G, -1)

    def process_edges(G, edges):
        for edge in nx.edge_bfs(G, -1):
            # simulate edge by edge
            # extract information
            leader, follower = edge

            segments = G.get_edge_data(leader, follower)["segments"]
            if leader == -1: # read from data directly
                for start, end in segments:
                    # # simulate follower's trajectory
                    # filtered_df = df[(df['Vehicle ID'] == follower) & (df['timestamps'] >= start) & (df['timestamps'] <= end)]
                    # filtered_df.mul(column_dict)
                    # filtered_df.rename(columns={'timestamps': 'Frame ID'}, inplace=True)
                    # # Reorder the columns to match the original DATA.txt file
                    # filtered_df = filtered_df[column_names]
                    # filtered_df.to_csv(write_file_name, sep='\t', index=False, header=False, mode="a")
                    try:
                        new_segments = union(G.nodes[follower]["segments"], [[start, end]])
                        # print(f"union of {G.nodes[follower]['segments']} and {[[start, end]]} is {new_segments}")
                    except Exception as e:
                        new_segments = union([], [[start, end]])
                    nx.set_node_attributes(G, {follower:{'segments':new_segments}})
            else:
                for start, end in segments:
                    leader_segments = intersection(G.nodes[leader]["segments"], [[start, end]])
                    if follower==710:
                        print(leader, leader_segments, G.nodes[leader]["segments"],  [[start, end]], G.nodes[710])
                    # print(f"intersection of {G.nodes[leader]['segments']} and {[[start, end]]} is {leader_segments}")
                    try:
                        new_segments = union(G.nodes[follower]["segments"], leader_segments)
                        # print(f"union of {G.nodes[follower]['segments']} and {leader_segments} is {new_segments}")
                    except KeyError:
                        new_segments = leader_segments
                    
                    # print(leader_segments, G.nodes[leader]["segments"])
                    
                    # new_segments = union()
                    nx.set_node_attributes(G, {follower:{'segments':new_segments}})
    
        return G
    
    # prioritize edges that are out_edges of nodes with no indegrees
    # the majority of nodes are in cycles - process them in BFS order
    edges_bfs = list(nx.edge_bfs(G, -1))
    edges_dfs = list(nx.edge_dfs(G, -1))
    H = copy.deepcopy(G)
    # find priority edges
    edges_priority = []
    while True:
        no_in_degree_nodes = [node for node in H.nodes if H.in_degree(node) == 0]
        if no_in_degree_nodes:
            edges_priority.extend(list(H.edges(no_in_degree_nodes)))
            H.remove_nodes_from(no_in_degree_nodes)
            # print(no_in_degree_nodes)
            # print("edges: ",len(edges))
        else:
            break
    # process priority edges first
    G = process_edges(G, edges_priority)
    # edges_remain = [edge for edge in edges_bfs+edges_bfs if edge not in edges_priority]
    G = process_edges(G, edges_bfs)
    G = process_edges(G, edges_dfs)
    
    # combined_df = pd.concat(filtered_dfs, ignore_index=True)
    # combined_df.to_csv(write_file_name, sep='\t', index=False, header=False)
    return G


if __name__ == "__main__":
    pair_id_file = "data/ngsim_pair_id_all_leaders.txt"
    read_file_name = 'data/DATA (NO MOTORCYCLES).txt'
    write_file_name = 'data/test.txt'
    G = fixed_leader_simulation("", pair_id_file, read_file_name,write_file_name)