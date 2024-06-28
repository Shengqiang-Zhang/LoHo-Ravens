import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

class PutBlocksOnZoneBisector(Task):
    """Arrange blocks between two designated zones on their bisector on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Arrange the {color} block between the zones on their bisector"
        self.task_completed_desc = "done arranging blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zones.
        zone_size = (0.12, 0.12, 0)
        zone_urdf = 'zone/zone.urdf'
        zone1_pose = self.get_random_pose(env, zone_size)
        rotation = utils.quatXYZW_to_eulerXYZ(zone1_pose[1])
        reverse_rotation = utils.eulerXYZ_to_quatXYZW((rotation[0], rotation[1], -rotation[2]))
        zone2_pose = self.get_random_pose(env, zone_size)
        zone2_pose = (zone2_pose[0],reverse_rotation)
        env.add_object(zone_urdf, zone1_pose, 'fixed')
        env.add_object(zone_urdf, zone2_pose, 'fixed')

        # Add blocks.
        n_block = 3
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for i in range(n_block):
            block_pose = self.get_random_pose(env, block_size)
            color = utils.get_random_color()[1][0]
            block_id = env.add_object(block_urdf, block_pose, color=color)
            blocks.append((block_id,color))

        # IMPORTANT Associate placement locations for goals.
        targs = self.place_blocks_along_bisector(zone1_pose, zone2_pose, n_block, gap=0.04)

        # Add goal
        for i in range(n_block):
            self.add_goal(objs=[blocks[i][0]], matches=np.ones((1, 1)), targ_poses=[targs[i]], replace=False,
                    rotations=True, metric='pose', params=None, step_max_reward=1/n_block, symmetries=[np.pi/2], 
                    language_goal=self.lang_template.format(color=blocks[i][1]))
        
    
    def place_blocks_along_bisector(self, pos1, pos2, n, gap=0.04, size=0.04):

        A = np.array(pos1[0])
        B = np.array(pos2[0])

        mid_point = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2]
        direction_vector = [(B[0] - A[0]), (B[1] - A[1])]
        length = np.linalg.norm(direction_vector)
        unit_direction = [direction_vector[0] / length, direction_vector[1] / length]
        normal_vector = [-unit_direction[1], unit_direction[0]]

        block_positions = []
        offset = (0.5*n-0.5)*gap + (n//2-1+0.5+n%2*0.5)*size
        distance = gap+size
        ini_position = np.array([mid_point[0] + offset * normal_vector[0], mid_point[1] + offset * normal_vector[1]])
        block_positions.append(ini_position)
        for i in range(n-1):   
            block_position = block_positions[i] + np.array([-distance * normal_vector[0], -distance * normal_vector[1]])
            block_positions.append(block_position)

        rotation = utils.eulerXYZ_to_quatXYZW((0,0,(utils.quatXYZW_to_eulerXYZ(pos1[1])[2]+utils.quatXYZW_to_eulerXYZ(pos2[1])[2])/2))
        block_positions=[((pos[0], pos[1], size/2),rotation) for pos in block_positions]

        return block_positions
