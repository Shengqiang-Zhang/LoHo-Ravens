import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class BuildConcentricCircles(Task):
    """Construct two distinct circles on the tabletop using 10 red and 10 blue blocks.
    Each circle should consist of blocks of the same color, with the blue circle larger and surrounding the red circle."""

    def __init__(self):
        super().__init__()
        self.max_steps = 30
        self.lang_template = "construct two concentric circles on the tabletop using 6 red and 10 blue blocks"
        self.task_completed_desc = "done building two circles."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        block_size = (0.04, 0.04, 0.04)
        # Add 6 red blocks.
        red_blocks = []
        for _ in range(6):
                block_pose = self.get_random_pose(env, obj_size=block_size)
                block_id = env.add_object('block/small.urdf', block_pose, color='red')
                red_blocks.append(block_id)
        radius = 0.1
        center = (0.45, 0, block_size[2] / 2)
        angles = np.linspace(0, 2 * np.pi, len(red_blocks), endpoint=False)

        # Define initial and target poses for the red circles.
        red_targ_poses = [((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), center[2]),
                       utils.eulerXYZ_to_quatXYZW((0,0,angle))) for angle in angles]

        # Add 10 blue blocks.
        blue_blocks = []
        for _ in range(10):
                block_pose = self.get_random_pose(env, obj_size=block_size)
                block_id = env.add_object('block/small.urdf', block_pose, color='blue')
                blue_blocks.append(block_id)
        radius = 0.2
        angles = np.linspace(0, 2 * np.pi, len(blue_blocks), endpoint=False)
        
        # Define initial and target poses for the blue circles.
        blue_targ_poses = [((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), center[2]),
                       utils.eulerXYZ_to_quatXYZW((0,0,angle))) for angle in angles]


        # Goal: each red block is in the red circle, each blue block is in the blue circle.
        self.add_goal(objs=red_blocks, matches=np.ones((6, 6)), targ_poses=red_targ_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2, language_goal='build red circle')
        self.add_goal(objs=blue_blocks, matches=np.ones((10, 10)), targ_poses=blue_targ_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2, language_goal='build blue circle')