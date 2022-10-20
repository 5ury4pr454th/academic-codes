import numpy as np

class SignalGenerator:

    def __init__(self, amplitude = 0.10, frequency = 0.001, y_offset = 0.00) -> None:
        self.amplitude = amplitude
        self.frequency = frequency
        self.time_period = 1.00/frequency
        self.y_offset = y_offset

    def constant(self, t) -> np.float64:
        """returns constant values"""
        return self.amplitude + self.y_offset
    
    def square(self, t) -> np.float64:
        """returns square waves"""
        if t % (self.time_period) <= 0.50 * (self.time_period):
            return self.amplitude + self.y_offset
        else:
            return -self.amplitude + self.y_offset

    def sawtooth(self, t) -> np.float64:
        """returns sawtooth waves"""
        if t % (self.time_period) <= 0.50 * (self.time_period):
            return (2*self.amplitude/self.time_period) * (t % (self.time_period)) + self.y_offset
        else:
            return (2*self.amplitude/self.time_period) * (t % (self.time_period)) + self.y_offset - 2*self.amplitude

    def step(self, t) -> np.float64:
        """returns step wave"""
        return self.amplitude + self.y_offset if t >= 0.0 else self.y_offset

    def random(self, t) -> np.float64:
        """random gaussian noise"""
        return np.random.normal(self.y_offset, self.amplitude)

    def sin(self, t) -> np.float64:
        """returns sine values"""
        return self.amplitude*np.sin(2*np.pi*self.frequency*t) + self.y_offset