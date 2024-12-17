import numpy as np

# def crawl(t):
#     s = t % 4.3 / 4.3
#     q = np.zeros(10,)
#     if s < 0.25:
#         q_roll = 30
#         q_yaw = 45
#     elif s >= 0.25 and s < 0.5:
#         q_roll = -30
#         q_yaw = 45
#     elif s >= 0.5:
#         q_roll = -30
#         q_yaw = -45
#     # print(s)
    
#     q[0], q[3] = q_roll, q_roll
#     q[1], q[4] = q_yaw, q_yaw
    
#     return q * np.pi / 180


def crawl_factory(breaks_roll, coefs_roll, breaks_yaw, coefs_yaw):
    """
    Perform PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) interpolation.
    
    Parameters:
    breaks : array-like
        Breakpoints (x-values) of the interpolation intervals
    coefs : array-like
        Coefficients for each polynomial segment
        Each row represents [a3, a2, a1, a0] for a cubic polynomial a3*x³ + a2*x² + a1*x + a0
    
    Returns:
    Function that can interpolate values at any point within the breaks
    """
    def interpolate(x):
        x = x % 4.3 / 4.3
        # Ensure x is a numpy array
        x = np.asarray(x)
        
        # Preallocate output array
        y = np.zeros_like(x, dtype=float)
        
        # Iterate through each interval
        for i in range(len(breaks_roll) - 1):
            # Find indices where x is in this interval
            mask = (x >= breaks_roll[i]) & (x <= breaks_roll[i+1])
            
            # Calculate local x (normalized to start of interval)
            local_x = x[mask] - breaks_roll[i]
            
            # Extract coefficients for this segment
            a3, a2, a1, a0 = coefs_roll[i]
            
            # Evaluate polynomial
            y[mask] = (a3 * local_x**3 + 
                       a2 * local_x**2 + 
                       a1 * local_x + 
                       a0)
            
        y2 = np.zeros_like(x, dtype=float)
        
        # Iterate through each interval
        for i in range(len(breaks_yaw) - 1):
            # Find indices where x is in this interval
            mask = (x >= breaks_yaw[i]) & (x <= breaks_yaw[i+1])
            
            # Calculate local x (normalized to start of interval)
            local_x = x[mask] - breaks_yaw[i]
            
            # Extract coefficients for this segment
            a3, a2, a1, a0 = coefs_yaw[i]
            
            # Evaluate polynomial
            y2[mask] = (a3 * local_x**3 + 
                       a2 * local_x**2 + 
                       a1 * local_x + 
                       a0)
        
        q = np.zeros(10,)
        q[0], q[3] = y, y
        q[1], q[4] = y2, -y2
        return q * np.pi/180


    return interpolate

# Example usage
# breaks = [0, 0.1, 0.25, 0.5, 0.6, 1]
# coefs = [
#     [0, 0, 0, 30],
#     [35555.5555555556, -8000.00000000000, 0, 30],
#     [0, 0, 0, -30],
#     [-120000.000000000, 18000.0000000000, 0, -30],
#     [0, 0, 0, 30]
# ]

# # Create interpolation function
# interpolator = pchip_interpolate(breaks, coefs)

# # Example of how to use the interpolator
# # You can interpolate single values or arrays
# print(interpolator(0.05))  # Interpolate at a single point
# print(interpolator([0.05, 0.2, 0.4]))  # Interpolate at multiple points
