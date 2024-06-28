import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class ConstructCircleWithBallInMiddle(Task):
    """Stack a set of small red blocks and small blue blocks in a circular shape on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "stack the {color} blocks in a circular shape"
        self.task_completed_desc = "done constructing circle with blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block colors and sizes.
        colors = ['red', 'blue']
        block_size = (0.04, 0.04, 0.04)

        # Add blocks.
        blocks = []
        for i in range(8):
            color=colors[i // 2 % 2]
            block_pose = self.get_random_pose(env, obj_size=block_size)
            block_id = env.add_object('block/block.urdf', block_pose, color=color)
            blocks.append(block_id)
        
        ball_size = (0.04, 0.04, 0.04)
        ball_pose = self.get_random_pose(env, obj_size=ball_size,bound=np.array([[0.25, 0.6], [0, 0.5], [0, 0.3]]))
        ball_id = env.add_object('ball/ball.urdf', ball_pose, color='gold')

        # Calculate target poses for circular shape.
        center = (0.45, 0.0, 0.02)
        radius = 0.15
        angles = np.linspace(0, 2 * np.pi, len(blocks), endpoint=False)
        targ_poses = [((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), center[2]),
                       utils.eulerXYZ_to_quatXYZW((0,0,angle))) for angle in angles]

        # Create language and motion goals for each block.
        for i, _ in enumerate(blocks):
            color = colors[i // 2 % 2]
            language_goal = self.lang_template.format(color=color)
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[targ_poses[i]], 
                          replace=False,rotations=True, metric='pose', params=None, step_max_reward=1 / len(blocks)/2,
                          language_goal=language_goal)
        self.add_goal(objs=[ball_id], matches=np.ones((1, 1)), targ_poses=[(center,(0,0,0,1))], 
                      replace=False,rotations=True, metric='pose', params=None, step_max_reward=1 / 2,
                      language_goal='Place the ball in the middle')