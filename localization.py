import sys
import numpy as np
from pymavlink import mavutil
import time
import threading
import os
from connect import connect_to_px4 , get_gps_data
connection_gps = connect_to_px4('udp:127.0.0.1:14550') 

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
def pixel_to_latlon(x, y, drone_lat, drone_lon):
    # Example conversion based on image resolution (adjust for actual resolution)
    lat_per_pixel = 0.000001  # Change this based on your system's scale (adjust for actual resolution)
    lon_per_pixel = 0.000001  # Change this based on your system's scale (adjust for actual resolution)

    # Convert pixel coordinates to geographical coordinates relative to the drone's GPS
    lat = drone_lat + (lat_per_pixel * (y - 540))  # Assuming the image height is 1080px, so center is at 540
    lon = drone_lon + (lon_per_pixel * (x - 960))  # Assuming the image width is 1920px, so center is at 960

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
    connection_string = 'udp:127.0.0.1:14550'
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
