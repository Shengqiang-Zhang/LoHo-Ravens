"""Stacking Block Pyramid Sequence task."""

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import pybullet as p
import random


class StackBlockPyramidSeqUnseenColors(Task):
    """Stacking Block Pyramid Sequence base class."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "put the {pick} block on {place}"
        self.task_completed_desc = "done stacking block pyramid."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: make bottom row.
        self.goals.append(([objs[0]], np.ones((1, 1)), [targs[0]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[0],
                                                         place="the lightest brown block"))

        self.goals.append(([objs[1]], np.ones((1, 1)), [targs[1]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[1],
                                                         place="the middle brown block"))

        self.goals.append(([objs[2]], np.ones((1, 1)), [targs[2]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[2],
                                                         place="the darkest brown block"))

        # Goal: make middle row.
        self.goals.append(([objs[3]], np.ones((1, 1)), [targs[3]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[3],
                                                         place=f"the {color_names[0]} and {color_names[1]} blocks"))

        self.goals.append(([objs[4]], np.ones((1, 1)), [targs[4]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[4],
                                                         place=f"the {color_names[1]} and {color_names[2]} blocks"))

        # Goal: make top row.
        self.goals.append(([objs[5]], np.ones((1, 1)), [targs[5]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[5],
                                                         place=f"the {color_names[3]} and {color_names[4]} blocks"))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlockPyramidSeqSeenColors(StackBlockPyramidSeqUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class StackBlockPyramidSeqFull(StackBlockPyramidSeqUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors


class StackBlockPyramidWithoutSeq(Task):
    """Stacking Block Pyramid without step-by-step instruction base class.
    There is just a high-level instruction: Stack blocks in a pyramid shape """

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "stack blocks in a pyramid shape"
        self.task_completed_desc = "done stacking block pyramid."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # self.goals.append((objs, np.eye(6), targs, False, True, 'pose', None, 1))
        # self.lang_goals.append(self.lang_template)
        # Goal: make bottom row.
        self.goals.append(([objs[0]], np.ones((1, 1)), [targs[0]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(pick=color_names[0],
        #                                                  place="the lightest brown block"))
        #
        self.goals.append(([objs[1]], np.ones((1, 1)), [targs[1]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(pick=color_names[1],
        #                                                  place="the middle brown block"))
        #
        self.goals.append(([objs[2]], np.ones((1, 1)), [targs[2]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(pick=color_names[2],
        #                                                  place="the darkest brown block"))
        #
        # # Goal: make middle row.
        self.goals.append(([objs[3]], np.ones((1, 1)), [targs[3]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(pick=color_names[3],
        #                                                  place=f"the {color_names[0]} and {color_names[1]} blocks"))
        #
        self.goals.append(([objs[4]], np.ones((1, 1)), [targs[4]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(pick=color_names[4],
        #                                                  place=f"the {color_names[1]} and {color_names[2]} blocks"))
        #
        # # Goal: make top row.
        self.goals.append(([objs[5]], np.ones((1, 1)), [targs[5]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(pick=color_names[5],
        #                                                  place=f"the {color_names[3]} and {color_names[4]} blocks"))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackAllBlock(Task):
    """Stacking ALL Blocks without step-by-step instruction base class.
    There is just a high-level instruction: Stack all blocks. """

    def __init__(self):
        super().__init__()
        self.n_blocks = random.randint(2, 5)
        self.max_steps = self.n_blocks + 2

        self.lang_template = "stack all the blocks"
        self.task_completed_desc = "done stacking block."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        self.scene_description = f"On the table, there are {self.n_blocks} blocks. " \
                                 f"Their colors are {color_names[:self.n_blocks]}. "

        # Associate placement locations for goals.
        # place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
        #              (0, 0.05, 0.03), (0, -0.025, 0.08),
        #              (0, 0.025, 0.08), (0, 0, 0.13)]
        place_pos = [(0, 0, 0.03 + 0.05 * i) for i in range(self.n_blocks)]
        # place_pos = [(0, 0, 0.03), (0, 0, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        match_matrix = np.eye(self.n_blocks)
        # match_matrix = np.ones((self.n_blocks, self.n_blocks))
        self.goals.append((objs, match_matrix, targs, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)
        # Goal: make bottom row.

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackAllBlockInAZone(Task):
    """Stacking ALL Blocks In A Zone without step-by-step instruction base class.
    There is just a high-level instruction: Stack all blocks in a zone. """

    def __init__(self):
        super().__init__()
        self.n_blocks = random.randint(2, 5)
        self.max_steps = self.n_blocks + 2

        self.lang_template = "stack all the blocks in the {zone_color} zone"
        self.task_completed_desc = "done stacking block."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]
        zone_selected_colors = [c for c in color_names[self.n_blocks:]]
        zone_color_ = random.sample(zone_selected_colors, 1)[0]
        zone_color = utils.COLORS[zone_color_]
        zone_size = (0.15, 0.15, 0)

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Add zones.
        zone_target = zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])

        self.scene_description = f"On the table, there are {self.n_blocks} blocks. " \
                                 f"Their colors are {color_names[:self.n_blocks]}. " \
                                 f"There is a zone on the table, and its color is {zone_color_}. "

        # Associate placement locations for goals.
        # place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
        #              (0, 0.05, 0.03), (0, -0.025, 0.08),
        #              (0, 0.025, 0.08), (0, 0, 0.13)]
        place_pos = [(0, 0, 0.03 + 0.05 * i) for i in range(self.n_blocks)]
        # place_pos = [(0, 0, 0.03), (0, 0, 0.08), (0, 0, 0.13)]
        # targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]
        targs = [(utils.apply(zone_pose, i), zone_pose[1]) for i in place_pos]

        self.goals.append((objs, np.eye(self.n_blocks), targs, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(zone_color=zone_color_))
        # Goal: make bottom row.

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackAllBlockOfSameColor(Task):
    """
    Stack ALL Blocks of the same color together.
    There are at most three kinds of color and at most 3 blocks with the same color.
   """

    def __init__(self):
        super().__init__()
        self.n_blocks = random.randint(3, 9)
        self.n_block_colors = random.randint(2, 3)
        self.max_steps = self.n_blocks + 2

        self.lang_template = "stack all the blocks of the same color together"
        self.task_completed_desc = "done stacking block."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose_list = []
        for i in range(self.n_block_colors):
            base_pose = self.get_random_pose(env, base_size)
            base_pose_list.append(base_pose)
            env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()
        random.shuffle(color_names)
        block_color_names = random.sample(color_names, self.n_block_colors)
        block_colors = [utils.COLORS[cn] for cn in block_color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_colors[i % self.n_block_colors] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        self.scene_description = f"On the table, there are {self.n_blocks} blocks, " \
                                 f"and their colors are {block_color_names}."

        # Associate placement locations for goals.
        # place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
        #              (0, 0.05, 0.03), (0, -0.025, 0.08),
        #              (0, 0.025, 0.08), (0, 0, 0.13)]
        # place_pos = [(0, 0, 0.03 + 0.05 * i) for i in range(self.n_blocks)]
        place_pos = [(0, 0, 0.03 + 0.05 * (i // self.n_block_colors)) for i in range(self.n_blocks)]
        # place_pos = [(0, 0, 0.03), (0, 0, 0.08), (0, 0, 0.13)]
        # targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]
        targs = [(utils.apply(base_pose_list[i % self.n_block_colors], pos),
                  base_pose_list[i % self.n_block_colors][1]) for i, pos in enumerate(place_pos)]

        match_matrix = np.eye(self.n_blocks)
        # match_matrix = np.ones((self.n_blocks, self.n_blocks))
        self.goals.append((objs, match_matrix, targs, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)
        # Goal: make bottom row.

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlockWithAlternateColor(Task):
    """Stack all blocks in the form of color crossed, starting from a specific color.
    There are always two kinds of color and at most 3 blocks with the same color.
   """
    # TODO: there are some bugs for creating the demonstrations for this task,
    #  ignore this task at the moment.

    def __init__(self):
        super().__init__()
        self.n_blocks = np.random.randint(2, 5)
        self.n_block_colors = 2
        self.max_steps = self.n_blocks + 2

        self.lang_template = "stack blocks with alternate colors"
        self.task_completed_desc = "done stacking block."

    def reset(self, env):
        super().reset(env)

        # Block colors.
        color_names = self.get_colors()
        block_color_names = random.sample(color_names, self.n_block_colors)
        block_colors = [utils.COLORS[cn] for cn in block_color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        first_block_pose = None
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_colors[i % self.n_block_colors] + [1])
            objs.append((block_id, (np.pi / 2, None)))
            if i == 0:
                first_block_pose = block_pose

        goal_poses = []
        for i in range(1, self.n_blocks):
            height = block_size[2] + block_size[2] * i
            goal_poses.append(
                (
                    (first_block_pose[0][0], first_block_pose[0][1], height),
                    first_block_pose[1]
                )
            )

        match_matrix = np.eye(self.n_blocks - 1)
        # match_matrix = np.ones((self.n_blocks, self.n_blocks))
        self.goals.append((objs[1:], match_matrix, goal_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(color=block_color_names[0]))
        # Goal: make bottom row.

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
