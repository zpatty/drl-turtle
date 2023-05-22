import numpy as np

class Environment:

    def __init__(self, low_location, high_location, obstructions):
        self.low_location = low_location
        self.high_location = high_location
        self.obstructions = obstructions

    def is_location_free(self, location, tolerance):
        distance_from_obstructions = np.linalg.norm(self.obstructions - location.reshape(3,1), axis=0)
        within_tolerance = distance_from_obstructions < tolerance
        return not within_tolerance.any()
    