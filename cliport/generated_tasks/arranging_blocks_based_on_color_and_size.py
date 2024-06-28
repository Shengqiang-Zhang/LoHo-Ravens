import os
import pybullet as p
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class ArrangingBlocksBasedOnColorAndSize(Task):
    """Pick up blocks of different colors (red, blue, green, yellow) and sizes (small, big) and arrange them on the tabletop in a specific pattern based on their color and size."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "arrange the blocks based on their color and size"
        self.task_completed_desc = "done arranging blocks"
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the colors and sizes of the blocks.
        colors = ['red', 'blue', 'green', 'yellow']
        sizes = ['small', 'big']

        # Shuffle the colors and sizes.
        np.random.shuffle(colors)
        np.random.shuffle(sizes)

        # Add blocks.
        objs = []
        for i in range(len(colors)):
            for j in range(len(sizes)):
                if sizes[j]=='small':
                    block_urdf = f'stacking/block.urdf'
                else:
                    block_urdf = f'stacking/bigger_block.urdf'
                block_pose = self.get_random_pose(env, (0.04, 0.04, 0.04))
                block_id = env.add_object(block_urdf, block_pose)
                p.changeVisualShape(block_id, -1, rgbaColor=utils.COLORS[colors[i]] + [1])
                objs.append((block_id, (np.pi / 2, None)))

        # Define the target poses for each color and size.
        targ_poses = []
        for i in range(len(colors)):
            for j in range(len(sizes)):
                x = 0.3 + (i * 0.1)
                y = 0.1 + (j * 0.2)
                z = 0.02
                pose = ((x, y, z), (0, 0, 0, 1))
                targ_poses.append(pose)

        # Create the match matrix for matching blocks with target poses.
        matches = np.eye(len(objs))

        self.add_goal(objs=objs, matches=matches, targ_poses=targ_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1,
                      language_goal=self.lang_template)