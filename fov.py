import math

def calculate_fov(focal_length_x, focal_length_y, sensor_width, sensor_height):
    fov_x = 2 * math.atan((sensor_width / 2) / focal_length_x)  # Horizontal FOV in radians
    fov_y = 2 * math.atan((sensor_height / 2) / focal_length_y)  # Vertical FOV in radians
    # If needed, convert FOV from radians to degrees
    # fov_x = fov_x * (180 / math.pi)
    # fov_y = fov_y * (180 / math.pi)
    
    print(f"Camera FOV: Horizontal = {fov_x:.2f} radians, Vertical = {fov_y:.2f} radians")
    return fov_x, fov_y

# Example usage:
focal_length_x = 2.1
focal_length_y = 2.1
sensor_width = 3.2  # mm
sensor_height = 2.6  # mm

calculate_fov(focal_length_x, focal_length_y, sensor_width, sensor_height)
