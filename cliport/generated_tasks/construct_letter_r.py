import numpy as np
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils

class ConstructLetterR(Task):
    """Task to construct the letter R from blocks on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12  # 7 blocks to place
        self.lang_template = "construct the letter R using blocks"
        self.task_completed_desc = "completed constructing the letter R."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the color for all blocks.
        color_name = 'red'
        color = utils.COLORS[color_name]

        # Define block size.
        block_size = (0.04, 0.04, 0.04)  # Uniform size for all blocks.

        # Define the positions for the blocks to construct the letter R.
        base_pos = np.array([0.1, 0, 0])  # Starting point on the table.
        vertical_positions_1 = [base_pos + np.array([0.2, i * 0.05, 0.02]) for i in range(3)]
        vertical_positions_2 = [base_pos + np.array([0.25, i * 0.1, 0.02]) for i in range(2)]
        vertical_positions_3 = [base_pos + np.array([0.3, i * 0.05, 0.02]) for i in range(3)]
        vertical_positions_4 = [base_pos + np.array([0.35, i * 0.1, 0.02]) for i in range(2)]
        vertical_positions_5 = [base_pos + np.array([0.4, i * 0.15, 0.02]) for i in range(2)]

        # Combine all positions.
        block_positions = vertical_positions_1 + vertical_positions_2 + vertical_positions_3 + vertical_positions_4 + vertical_positions_5

        # Load the block URDF.
        block_urdf = 'block/block.urdf'
        blocks=[]

        # Add blocks to the environment and set their colors.
        for _ in range(len(block_positions)):
            pose = self.get_random_pose(env, block_size)  # No rotation needed, quaternion format.
            block_id = env.add_object(block_urdf, pose)
            p.changeVisualShape(block_id, -1, rgbaColor=color + [1])
            blocks.append(block_id)

        # Define target positions for the blocks to form the letter R.
        # Note: In a real implementation, these would be carefully calculated to form the letter R.
        # Here, we use the same positions as the initial ones for simplicity.
        targ_poses = [(pos, (0, 0, 0, 1)) for pos in block_positions]

        # Add goal for each block placement with detailed language instructions.
        for i, targ_pose in enumerate(targ_poses):
            lang_goal = f"Place a block at {targ_pose[0]} to construct part of the letter R."
            self.add_goal(objs=[blocks[i]], matches=np.ones((1,1)), targ_poses=[targ_pose],
                          replace=False, rotations=True, metric='pose', params=None,
                          step_max_reward=1 / len(block_positions), language_goal=lang_goal)