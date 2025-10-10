import math
import numpy as np

def y_to_bin_index(y, offset_interval, y_max):
    #round y to interval of 0.25 and then convert it to its respective bin index
    y_clamped = np.clip(y, -y_max, y_max)
    normalize = y_clamped + y_max
    nearest = round(normalize / offset_interval) * offset_interval
    return int(nearest / offset_interval)

def yaw_to_bin_index(yaw, yaw_interval):
    #round y to interval of pi/6 and then give it the respective bin

    #yaw is capped -pi to pi already
    yaw_shifted = yaw + math.pi
    nearest = round(yaw_shifted / yaw_interval) * yaw_interval
    bin_index = int(nearest / yaw_interval)
    return bin_index

def state_to_bin_index(y, offset_interval, y_max, num_yaw_bins, yaw, yaw_interval):
    y_bin = y_to_bin_index(y, offset_interval, y_max)
    yaw_bin = yaw_to_bin_index(yaw, yaw_interval)
    return int(y_bin*num_yaw_bins + yaw_bin)

print(y_to_bin_index(-4, 0.25, 4))
print(yaw_to_bin_index(-1*math.pi/6, math.pi/6))

print(state_to_bin_index(-4, 0.25, 4, 2*math.pi/(math.pi/6), -6*math.pi/6, math.pi/6))