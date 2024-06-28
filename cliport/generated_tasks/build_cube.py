import numpy as np
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils

class BuildCubeWithSameColorBlock(Task):
    """
    Task to construct a cube structure using six blocks of the same color.
    The blocks are first identified and gathered from around the table,
    then accurately stacked to form a cube without any external support.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "build a cube with {color} blocks"
        self.task_completed_desc = "done building cube with same color blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block size and color
        block_size = (0.04, 0.04, 0.04)  # Size of each block
        color_rgb, color_name = utils.get_random_color()  # Get a random color for the blocks
        color_rgb=color_rgb[0]
        color_name=color_name[0]

        # Add eight blocks of the same color
        block_urdf = 'block/block.urdf'
        blocks = []
        for _ in range(8):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=color_rgb + [1])
            blocks.append(block_id)

        # Define target positions for the cube structure
        # Assuming the center of the table is at [0.5, 0, 0] and blocks are 0.04m in size
        base_height = 0.02  # Half of block's height
        positions = [
            (0.35, -0.05, base_height), (0.42, -0.05, base_height),  # Bottom layer
            (0.35, 0.02, base_height), (0.42, 0.02, base_height),
            (0.35, -0.05, base_height + 0.04), (0.42, -0.05, base_height + 0.04),  # Top layer
            (0.35, 0.02, base_height + 0.04), (0.42, 0.02, base_height + 0.04)
        ]
        targ_poses = [(pos, (0, 0, 0, 1)) for pos in positions]  # No rotation

        # Add goals for each block to be in the correct position
        language_goal = self.lang_template.format(color=color_name)
        self.add_goal(objs=blocks, matches=np.eye(8), targ_poses=targ_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1, language_goal=language_goal)