import traci
import traci.constants as tc
import xml.etree.ElementTree as ET
# import pandas as pd

def run_simulation():
    traci.start(["sumo", "-c", "demo.sumocfg"])

    # Run simulation steps
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

    traci.close()


def get_vehicle_ids_from_routes(route_file):
    tree = ET.parse(route_file)
    root = tree.getroot()

    vehicle_ids = []
    for route in root.findall('.//vehicle'):
        vehicle_id = route.get('id')
        vehicle_ids.append(vehicle_id)

    return vehicle_ids





def write_vehicle_trajectories_to_csv(filename):
    # Start SUMO simulation with TraCI
    traci.start(["sumo", "-c", "demo.sumocfg"])
    
    # Replace "your_routes_file.rou.xml" with the actual path to your SUMO route file
    route_file_path = "demo.rou.xml"
    # Get a list of vehicle IDs from the route file
    predefined_vehicle_ids = get_vehicle_ids_from_routes(route_file_path)

    # Print the list of vehicle IDs
    print("List of Predefined Vehicle IDs:", predefined_vehicle_ids)


    # Open the CSV file for writing
    with open(filename, 'w') as csv_file:
        # Write header
        csv_file.write("Time,VehicleID,PositionX,PositionY,Speed\n")

        # Run simulation steps
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            # Get simulation time
            simulation_time = traci.simulation.getTime()

            # Get IDs of all vehicles
            vehicle_ids = traci.vehicle.getIDList()

            # Iterate over all vehicles
            for vehicle_id in vehicle_ids:
                # Get vehicle position and speed
                position = traci.vehicle.getPosition(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)

                # Write data to the CSV file
                csv_file.write(f"{simulation_time},{vehicle_id},{position[0]},{position[1]},{speed}\n")

            # Simulate one step
            traci.simulationStep()
            step += 1
            # if (step%100 == 0):
            #     print("step: ", step)
            if step > 1000:
                break
    # Close connection
    traci.close()
    print("Complete!")

    return

if __name__ == "__main__":
    # Uncomment and use one of the functions based on your needs
    # run_simulation()
    write_vehicle_trajectories_to_csv('vehicle_trajectories.csv')

