import numpy as np
import os
import pybullet as p
import random
from cliport.tasks.task import Task
from cliport.utils import utils

class InsertBlocksToDifferentColorFixture(Task):
    """Pick up blocks and insert them into the different colored fixtures."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert the block with different colors into the {color} fixture"
        self.task_completed_desc = "done inserting blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for blocks and fixtures
        colors = ['red', 'blue', 'silver', 'gold']

        # Add fixtures.
        fixture_size = (0.04, 0.04, 0.04)
        fixture_urdf = 'insertion/fixture.urdf'
        fixture_poses = []
        for i in range(len(colors)):
            fixture_pose = self.get_random_pose(env, fixture_size)
            fixture_id = env.add_object(fixture_urdf, fixture_pose, color=colors[i], category='fixed')
            fixture_set_poses = [fixture_pose,
                                (utils.apply(fixture_pose, (0.04, 0, 0)), fixture_pose[1]),
                                (utils.apply(fixture_pose, (0, 0.04, 0)), fixture_pose[1])]
            fixture_poses.append(fixture_set_poses)

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for k in range(len(colors)):
            blocks.append([])
            for i in range(3):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[colors[k]])
                blocks[k].append(block_id)

        # Goal: each block is in the corresponding color fixture.
        for i, color in enumerate(colors):
            temp_blocks=[]
            for j in range(len(colors)):
                if j != i:
                    temp_blocks.append(blocks[j][-1])
                    blocks[j]=blocks[j][:-1]
            temp_poses = fixture_poses[i].copy()
            random.shuffle(temp_poses)
            self.add_goal(objs=temp_blocks, matches=np.eye(3), targ_poses=temp_poses, replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1/len(colors),
                          language_goal=self.lang_template.format(color=color))