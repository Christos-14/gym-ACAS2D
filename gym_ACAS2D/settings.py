from scipy.constants import g


# WINDOW CONSTANTS
WIDTH = 600
HEIGHT = 600
FPS = 1000
CAPTION = "ACAS-2D"
FONT_NAME = "freesansbold.ttf"
FONT_SIZE = 14
FONT_RGB = (0, 0, 0)
SKY_RGB = (204, 204, 255)
GREEN_RGB = (0, 255, 0)
RED_RGB = (255, 0, 0)

# AIRCRAFT CONSTANTS
MIN_TRAFFIC = 0
MAX_TRAFFIC = 8
# N_TRAFFIC = 4
AIRCRAFT_SIZE = 24  # images used are 24x24 pixels
COLLISION_RADIUS = 48

# KINEMATICS CONSTANTS
AIRSPEED = 0.1
AIRSPEED_FACTOR_MIN = 0.8
AIRSPEED_FACTOR_MAX = 1.2
ACC_LAT_LIMIT = 2 * g
MIN_BOUNCE_ANGLE = 120
MAX_BOUNCE_ANGLE = 240

# GAME CONSTANTS
EPISODES = 10
MAX_STEPS = 10000
OUTCOME_NAMES = {1: 'Goal', 2: 'Collision', 3: 'Timeout'}

# REWARD CONSTANTS
REWARD_GOAL = 1000
REWARD_COLLISION = -1000
REWARD_STEP = -0.01
REWARD_MIN_SEPARATION_FACTOR = 0.001

# IMAGE FILES
LOGO = "png/004-compass.png"
PLAYER_IMG = "png/001-plane.png"
TRAFFIC_IMG = "png/002-travelling.png"
GOAL_IMG = "png/003-army.png"
