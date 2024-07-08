"""
Read NGSIM csv file
Extract Leader-follower pairs
Write pair id list into a .txt file
    [follower_id, leader_id, event_start_time, event_end_time]
"""
import os

def write_pair_ids(data_file, write_file_name="pair_id.txt"):

    # initialize
    prev_vehicle_id = None
    prev_leader_id = -1
    prev_time = 0 # time of the previous line
    start_time = 0 # time of start of the current event

    # Read the data file line by line
    # Open input and output TXT files
    with open(data_file, mode='r') as input_file, \
        open(write_file_name, mode='w') as output_file:

        for line in input_file:
            # Split the line into columns
            columns = line.strip().split()

            # extract current info
            vehicle_id = int(columns[0])
            leader_id = int(columns[9])
            time = float(columns[1]) * 0.1

            # Check if current [foll- lead pair changes], trigger a new event
            if vehicle_id != prev_vehicle_id or leader_id != prev_leader_id:

                # start a new pair
                output_row = [prev_vehicle_id, prev_leader_id, start_time, prev_time]

                # writing condition: if leader is valid
                # if prev_leader_id != -1:
                line = " ".join(str(item) for item in output_row) + '\n'
                output_file.write(line)
                print(output_row)

                # update event start time for next
                start_time = time

            # update current line
            prev_vehicle_id = vehicle_id
            prev_leader_id = leader_id
            prev_time = time

       
    print(f"The last follower-leader pair is (not written to {write_file_name}): ")
    print([prev_vehicle_id, prev_leader_id, start_time, prev_time])

    print("complete!")



def write_od_trips(data_file, write_file_name="ngism_od_trips.txt"):
    """
    For felipe to start SUMO
    """
    # initialize
    prev_vehicle_id = None
    # prev_time = 0 # time of the previous line
    start_time = 0 # time of start of the current event

    # Read the data file line by line
    # Open input and output TXT files
    with open(data_file, mode='r') as input_file, \
        open(write_file_name, mode='w') as output_file:

        for line in input_file:
            # Split the line into columns
            columns = line.strip().split()

            # extract current info
            vehicle_id = int(columns[0])
            time = float(columns[1]) * 0.1
            pos = float(columns[3])
            speed = float(columns[4])

            # Check if current [foll- lead pair changes], trigger a new event
            if vehicle_id != prev_vehicle_id:

                if prev_vehicle_id is not None:
                    # start a new pair
                    output_row = [prev_vehicle_id, start_pos, prev_pos, start_time, start_speed] # vehID, start pos, end pos, start time, start speed

                    line = " ".join(str(item) for item in output_row) + '\n'
                    print(line)
                    output_file.write(line)

                # record the starting info of the next trip
                start_time = time
                start_speed = speed
                start_pos = pos

            # record the ending info of this trip
            prev_vehicle_id = vehicle_id
            # prev_time = time
            prev_pos = pos

    print("complete!")



def get_lane_change(data_file, write_file_name="lane_change.txt"):
    '''
    give felipe for microsim
    get teh exact time of lane change for each vehicle
    vehID, time, from_lane, to_lane
    from_lane is -1 if this trajectory just gets started
    '''
    # initialize
    prev_vehicle_id = None
    prev_leader_id = -1
    prev_time = 0 # time of the previous line
    start_time = 0 # time of start of the current event


    # Read the data file line by line
    # Open input and output TXT files
    with open(data_file, mode='r') as input_file, \
        open(write_file_name, mode='w') as output_file:

        for line in input_file:
            # Split the line into columns
            columns = line.strip().split()

            # extract current info
            vehicle_id = int(columns[0])
            time = float(columns[1]) * 0.1
            lane = int(columns[2])

            
            if vehicle_id == prev_vehicle_id:
            
                # lane change
                if lane != from_lane:

                    # start a record
                    output_row = [vehicle_id, time, from_lane, lane]
                    line = " ".join(str(item) for item in output_row) + '\n'
                    output_file.write(line)
                    print(output_row)

            else: # new vehicle
                from_lane = -1
                output_row = [vehicle_id, time, from_lane, lane]
                line = " ".join(str(item) for item in output_row) + '\n'
                output_file.write(line)
                print(output_row)


            # update current line
            prev_vehicle_id = vehicle_id
            from_lane = lane

    print("complete!")



if __name__ == "__main__":
    # Path to your data file
    data_path = "data/"
    file_name = "DATA (NO MOTORCYCLES).txt"
    # write_file_name = os.path.join(data_path, "ngism_od_trips_.txt")
    data_file = os.path.join(data_path, file_name)
    # write_pair_ids(data_file, write_file_name="ngsim_pair_id_all_leaders.txt")

    # write_od_trips(data_file, write_file_name)
    get_lane_change(data_file, write_file_name="ngsim_lane_change.txt")

