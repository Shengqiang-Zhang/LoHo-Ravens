import numpy as np
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils

class ConstructLetterM(Task):
    """Construct the letter M using blocks and boxes with specific colors."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10  # Define the maximum steps allowed to complete the task.
        self.lang_template = "construct the letter M using red, blue, and green blocks and boxes"
        self.task_completed_desc = "constructed letter M."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for the task.
        colors = {
            'red': utils.COLORS['red'],
            'blue': utils.COLORS['blue'],
            'green': utils.COLORS['green']
        }

        # Define sizes for blocks and boxes.
        block_size = (0.05, 0.05, 0.05)  # Size for small blocks.
        box_size = (0.04, 0.04, 0.04)  # Size for boxes to be modified to form the M shape.

        # Add objects to the environment.
        objs = []
        positions = [
            (0.45, -0.1, 0.025),  # Left base of M
            (0.45, 0.1, 0.025),   # Right base of M
            (0.45, -0.05, 0.125), # Left ascending part of M
            (0.45, 0.05, 0.125),  # Right ascending part of M
            (0.45, 0, 0.225)      # Middle top of M
        ]
        colors_order = ['red', 'blue', 'green', 'blue', 'red']  # Color order for the parts of M.

        # Add modified boxes to form the M shape.
        for i, pos in enumerate(positions):
            if i in [2, 3]:  # Modify size for ascending parts.
                modified_size = (box_size[0], box_size[1], box_size[2] * 2)
            else:
                modified_size = box_size
            box_urdf = self.fill_template('box/box-template.urdf', {'DIM': modified_size})
            box_pose = self.get_random_pose(env, modified_size)
            box_id = env.add_object(box_urdf, box_pose)
            p.changeVisualShape(box_id, -1, rgbaColor=colors[colors_order[i]] + [1])
            objs.append((box_id, (np.pi / 2, None)))  # Assuming symmetry for rotation.

        # Define target positions for each part of the M to construct the letter correctly.
        # These positions are slightly adjusted from the initial positions to ensure the letter M is constructed properly.
        targs = [((pos[0], pos[1], pos[2] + 0.05), (0, 0, 0, 1)) for pos in positions]

        # Add goals for each step of constructing the letter M.
        for i in range(len(objs)):
            self.add_goal(objs=[objs[i]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / len(objs),
                          symmetries=[np.pi/2], language_goal=self.lang_template)

        # Set the scene description for the task.
        self.scene_description = "On the table, there are blocks and boxes to construct the letter M. " \
                                 "Use the red, blue, and green colored assets for visual distinction and spatial reasoning."