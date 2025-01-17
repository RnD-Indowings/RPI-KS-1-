import asyncio
from mavsdk import System

# Function to print the current altitude
async def print_current_altitude(drone):
    async for position in drone.telemetry.position():
        current_alt = position.relative_altitude_m  # Altitude relative to takeoff point
        print(f"Current altitude: {current_alt:.2f} meters")

# Function to get and print the yaw angle
async def get_yaw_angle(drone):
    async for euler_angle in drone.telemetry.attitude_euler():
        yaw = euler_angle.yaw_deg
        print(f"Yaw Angle: {yaw:.2f} degrees")

async def run():
    drone = System()
    #await drone.connect(system_address="serial:///dev/ttyACM0:1152000")
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break

    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break

    # Start altitude and yaw angle monitoring tasks
    altitude_task = asyncio.ensure_future(print_current_altitude(drone))
    yaw_task = asyncio.ensure_future(get_yaw_angle(drone))

    # Ensure the program keeps running
    await asyncio.gather(altitude_task, yaw_task)

if __name__ == "__main__":
    asyncio.run(run())
