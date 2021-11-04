from scipy.constants import g

# GAME CONSTANTS
OUTCOME_NAMES = {1: 'Goal', 2: 'Collision', 3: 'Timeout'}
TEST_EPISODES = 1000
EVAL_EPISODES = 10
MAX_STEPS = 1200
N_STEPS = 2048
TOTAL_STEPS = N_STEPS * 512  # 1,048,576
EVAL_STEPS = TOTAL_STEPS / 32

# WINDOW CONSTANTS
WIDTH = 1000
HEIGHT = 800
FPS = 100
CAPTION = "ACAS-2D"
FONT_NAME = "freesansbold.ttf"
FONT_SIZE = 14
BLACK_RGB = (0, 0, 0)
SKY_RGB = (60, 150, 220)
GREEN_RGB = (0, 255, 0)
RED_RGB = (255, 0, 0)
YELLOW_RBG = (255, 255, 0)

# RANDOM SEED
RANDOM_SEED = 13

# AIRCRAFT CONSTANTS
MIN_TRAFFIC = 1
MAX_TRAFFIC = 1
AIRCRAFT_SIZE = 24  # images used are 24x24 pixels
COLLISION_RADIUS = 2 * AIRCRAFT_SIZE
GOAL_RADIUS = 4 * AIRCRAFT_SIZE
SAFE_DISTANCE = 4 * COLLISION_RADIUS

# KINEMATICS CONSTANTS
AIRSPEED = 120
AIRSPEED_FACTOR_MIN = 1
AIRSPEED_FACTOR_MAX = 1
ACC_LAT_LIMIT = 2400 * g
PLAYER_INITIAL_HEADING_LIM = 5
TRAFFIC_INITIAL_HEADING_LIM = 20

# REWARD CONSTANTS
REWARD_GOAL = 1000
REWARD_COLLISION = -1000
# REWARD_TIMEOUT = -1000

# IMAGE FILES
LOGO = "png/004-compass.png"
PLAYER_IMG = "png/001-plane.png"
TRAFFIC_IMG = "png/002-travelling.png"
GOAL_IMG = "png/003-army.png"
