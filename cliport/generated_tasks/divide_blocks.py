import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class DivideBlocks(Task):
    """Divide blocks into several groups of three."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "divide the blocks into groups of three"
        self.task_completed_desc = "done dividing blocks into groups"
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Generate random number of blocks.
        n_blocks = np.random.randint(4, 10)

        # Generate random colors for the blocks.
        selected_color_names = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray']
        colors = [utils.COLORS[c] for c in selected_color_names]

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=colors[i % len(colors)])
            blocks.append(block_id)

        # Divide blocks into groups of three.
        groups = []
        for i in range(0, n_blocks, 3):
            if i+3 <= n_blocks:
                group = blocks[i:i+3]
                groups.append(group)
            else:
                group = blocks[i:]
                groups.append(group)    

        group_targs=[self.get_random_pose(env,block_size*3) for _ in range(len(groups))]
        targs = [[((targ[0][0], targ[0][1], targ[0][2]+block_size[2]*i),(0,0,0,1)) for i in range(3)] for targ in group_targs[:-1]]
        last_targ=[((group_targs[-1][0][0],group_targs[-1][0][1],group_targs[-1][0][2]+block_size[2]*i),(0,0,0,1)) for i in range(n_blocks-3*len(groups)+3)]

        # Goal: divide blocks into groups of three.
        language_goal = self.lang_template
        '''
        for group in groups:
            self.add_goal(objs=group, 
                          matches=np.ones((3,3)) if groups.index(group) < len(groups)-1 else np.ones((n_blocks-3*len(groups)+3,n_blocks-3*len(groups)+3)), 
                          targ_poses=targs[groups.index(group)] if groups.index(group) < len(groups)-1 else last_targ, 
                          replace=False, rotations=True, metric='pose', params=None, step_max_reward=1/len(groups), symmetries=[np.pi/2],
                          language_goal=language_goal)
        '''
        for i, group in enumerate(groups):
            for j, block_id in enumerate(group):
                self.add_goal(objs=[block_id], matches=np.eye(1), 
                              targ_poses=[targs[i][j]] if i<len(groups)-1 else [last_targ[j]], 
                              replace=True, rotations=True, 
                              metric='pose', params=None, step_max_reward=1/n_blocks, symmetries=[np.pi/2],
                              language_goal=language_goal)