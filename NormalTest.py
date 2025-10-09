import math

def getIdealYaw(offset, lookahead=5):
    return math.atan2(offset, math.sqrt(math.pow(lookahead, 2) - math.pow(offset, 2)))

print(getIdealYaw(6))
