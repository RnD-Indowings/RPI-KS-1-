#!/usr/bin/env python3


import asyncio

from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed)


async def run():
    """ Does Offboard control using velocity body coordinates. """

    drone = System()
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

    print("-- Arming")
    await drone.action.arm()

    print("-- Setting initial setpoint")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: \
              {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    print("-- Fly the square pattern")

    # First side of the square
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(5.0, 0.0, -10.0, 0.0))  # Move forward
    await asyncio.sleep(5)  # Move for 5 seconds (adjust as needed)

    # Turn 90 degrees (clockwise)
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 90.0))
    await asyncio.sleep(1)  # Time to turn 90 degrees

    # Second side of the square
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(5.0, 0.0, 0.0, 0.0))  # Move forward
    await asyncio.sleep(5)

    # Turn 90 degrees (clockwise)
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 90.0))
    await asyncio.sleep(1)

    # Third side of the square
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(5.0, 0.0, 0.0, 0.0))  # Move forward
    await asyncio.sleep(5)

    # Turn 90 degrees (clockwise)
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 90.0))
    await asyncio.sleep(1)

    # Fourth side of the square
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(5.0, 0.0, 0.0, 0.0))  # Move forward
    await asyncio.sleep(5)

    # Turn 90 degrees (clockwise) to complete the square
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 90.0))
    await asyncio.sleep(1)

    print("-- Wait for a bit")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: \
              {error._result.result}")
        
        
    
    print("RTL")
    await drone.mission.set_return_to_launch(True)




if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())