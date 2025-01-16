import sys
import numpy as np
from pymavlink import mavutil
import time
import threading
import os
from connect import connect_to_px4 , get_gps_data
connection_gps = connect_to_px4('127.0.0.1:14550') 

terminal_opened = False
lock = threading.Lock()
def run_program_after_delays(lat , lon):
    global terminal_opened
    with lock:
        if terminal_opened:
            return
        terminal_opened = True
        time.sleep(15)
    
    # Run the program using os.system with arguments
    command = f'gnome-terminal -- bash -c "python3 kinetic_strike.py  {lat} {lon}; exec bash"'
    
    # Execute the command to open the terminal and run the script
    os.system(command)
    run_program_after_delays(lat, lon)
    return

# Function to get GPS data from PX4
def get_gps_data(connection):
    connection.mav.request_data_stream_send(connection.target_system, connection.target_component, mavutil.mavlink.MAV_DATA_STREAM_POSITION, 1, 1)
    msg = connection.recv_match(type='GPS_RAW_INT', blocking=True)
    if msg:
        lat = msg.lat / 1e7  # Convert raw GPS data to degrees
        lon = msg.lon / 1e7
        return lat, lon
    else:
        return None, None

# Function to convert pixel coordinates to latitude and longitude

def pixel_to_latlon(x, y, drone_lat, drone_lon, altitude, camera_fov_horizontal, camera_fov_vertical, img_width, img_height):

    camera_fov_horizontal = 0.65
    camera_fov_vertical  = 0.49
    img_height = 640
    img_width = 480
    """
    Convert pixel coordinates to latitude and longitude dynamically.

    Parameters:
        x, y: Pixel coordinates.
        drone_lat, drone_lon: Current GPS coordinates of the drone.
        altitude: Current altitude of the drone in meters.
        camera_fov_horizontal: Camera's horizontal field of view in degrees.
        camera_fov_vertical: Camera's vertical field of view in degrees.
        img_width, img_height: Camera resolution in pixels.

    Returns:
        Converted latitude and longitude.
    """
    # Convert FoV from degrees to radians
    fov_horizontal_rad = np.radians(camera_fov_horizontal)
    fov_vertical_rad = np.radians(camera_fov_vertical)

    # Calculate the ground coverage (width and height in meters) at the given altitude
    ground_width = 2 * altitude * np.tan(fov_horizontal_rad / 2)
    ground_height = 2 * altitude * np.tan(fov_vertical_rad / 2)

    # Calculate meters per pixel
    meters_per_pixel_x = ground_width / img_width
    meters_per_pixel_y = ground_height / img_height

    # Calculate the offset in meters from the image center
    offset_x_meters = (x - img_width / 2) * meters_per_pixel_x
    offset_y_meters = (y - img_height / 2) * meters_per_pixel_y

    # Approximate conversion from meters to degrees (latitude and longitude)
    # 1 degree latitude = ~111,111 meters
    # 1 degree longitude = ~111,111 * cos(latitude) meters
    meters_per_degree_lat = 111111
    meters_per_degree_lon = meters_per_degree_lat * np.cos(np.radians(drone_lat))

    # Convert the offsets in meters to offsets in latitude and longitude
    offset_lat = offset_y_meters / meters_per_degree_lat
    offset_lon = offset_x_meters / meters_per_degree_lon

    # Calculate the final latitude and longitude
    lat = drone_lat + offset_lat
    lon = drone_lon + offset_lon

    return lat, lon


# Main function to process the command-line arguments and get GPS data
def main():
    if len(sys.argv) != 3:
        print("Usage: python mdetect.py <centre_x> <centre_y>")
        sys.exit(1)

    # Access the command-line arguments
    center_x = int(sys.argv[1])  # Argument 1
    center_y = int(sys.argv[2])  # Argument 2
    print(f"Received Pixel: ({center_x}, {center_y})")

    # Connect to PX4
    connection_string = '127.0.0.1:14550'
    pixhawk_connection = connect_to_px4(connection_string)
    if pixhawk_connection:
        while True:
            # Get the current GPS position of the drone
            latitude, longitude = get_gps_data(pixhawk_connection)
            if latitude is not None and longitude is not None:
                print(f"Current GPS Coordinates: Latitude = {latitude}, Longitude = {longitude}")

                # Convert the pixel coordinates to lat/lon
                lat, lon = pixel_to_latlon(center_x, center_y, latitude, longitude)
                print(f"Converted GPS: Latitude = {lat}, Longitude = {lon}")
            else:
                print("Unable to retrieve GPS coordinates.")
            
            # Add a small delay to avoid flooding the output
            time.sleep(1)
            if not terminal_opened:
                thread = threading.Thread(target=run_program_after_delays, args=(lat, lon))
                thread.start()

if __name__ == '__main__':
    
    main()
