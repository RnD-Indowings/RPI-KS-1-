import os
from pymavlink import mavutil
connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')


def execute_script():
    script_path = "mdetect.py"  
    os.system(f"python {script_path}")
CHANNEL_15_THRESHOLD = 1500 
print("Listening for RC channel 15 activation...")

while True:
    message = connection.recv_match(blocking=True)
    if message:
        message_type = message.get_type()

        if message_type == "RC_CHANNELS":
            channel_15_value = message.chan15_raw 
            #print(f"Channel 15 value: {channel_4_value}")


            if channel_15_value > CHANNEL_15_THRESHOLD:
                print("Channel 15 activated! Executing script...")
                execute_script()
                while channel_15_value > CHANNEL_15_THRESHOLD:
                    message = connection.recv_match(blocking=True)
                    if message.get_type() == "RC_CHANNELS":
                        channel_15_value = message.chan15_raw

