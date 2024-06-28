import random
from typing import List, Set, Union

import numpy as np
import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class PickAndPlacePrimitive(Task):
    """
    Pick-and-place primitive for the LLM planner.
    This primitive is trained with all the colors, following the setting of `Inner Monologue`.
    Pick up the [block1] and place it on the [block2/bowl/zone].
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = ("pick up the {pick_color} block "
                              "and place it on the {place_color} {place}.")
        self.task_completed_desc = "done placing blocks."

    def reset(self, env):
        super().reset(env)
        target_idx = random.randint(0, 2)
        target_objs = ["block", "bowl", "zone"]
        all_color_names = self.get_colors()

        n_blocks = np.random.randint(2, 4)
        n_zones = np.random.randint(1, 3)
        n_bowls = np.random.randint(2, 4)

        block_colors = random.sample(all_color_names, n_blocks)
        bowl_colors = random.sample(all_color_names, n_bowls)
        zone_colors = random.sample(all_color_names, n_zones)
        block_util_colors = [utils.COLORS[cn] for cn in block_colors]
        bowl_util_colors = [utils.COLORS[cn] for cn in bowl_colors]
        zone_util_colors = [utils.COLORS[cn] for cn in zone_colors]

        self.scene_description = (f"On the table, there are {n_blocks} blocks. "
                                  f"Their colors are {block_colors}. "
                                  f"There are {n_zones} zones. Their colors are {zone_colors}. "
                                  f"There are {n_bowls} bowls. Their colors are {bowl_colors}. ")

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        target_bowl_pose = None
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_util_colors[i] + [1])
            if i == 0:
                target_bowl_pose = bowl_pose

        # Add blocks.
        blocks = []
        block_pts = {}
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        pick_block = None
        place_pose = None
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_util_colors[i] + [1])
            if i == 0:
                block_pts[block_id] = self.get_box_object_points(block_id)
                if target_idx == 1:
                    pick_block = (block_id, (0, None))
                else:
                    pick_block = (block_id, (np.pi / 2, None))
            elif i == 1:
                place_pose = block_pose

        # Add zones.
        zone_size = (0.1, 0.1, 0)
        target_zone_pose = None
        for i in range(n_zones):
            zone_pose = self.get_random_pose(env, zone_size)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_util_colors[i] + [1])
            if i == 0:
                target_zone_pose = zone_pose

        if target_objs[target_idx] == "block":
            target_pose = (
                (place_pose[0][0], place_pose[0][1], block_size[2] + block_size[2] / 2),
                place_pose[1]
            )
            self.goals.append(
                ([pick_block], np.eye(1), [target_pose], False, True, 'pose', None, 1)
            )
            self.lang_goals.append(self.lang_template.format(pick_color=block_colors[0],
                                                             place_color=block_colors[1],
                                                             place=target_objs[target_idx]))
        elif target_objs[target_idx] == "bowl":
            self.consider_z_in_match = False
            target_matrix = np.ones((1, 1))
            self.goals.append(
                ([pick_block], target_matrix, [target_bowl_pose], False, True, 'pose', None, 1)
            )
            self.lang_goals.append(self.lang_template.format(pick_color=block_colors[0],
                                                             place_color=bowl_colors[0],
                                                             place=target_objs[target_idx]))
        elif target_objs[target_idx] == "zone":
            target_matrix = np.ones((1, 1))
            pos, rot = target_zone_pose
            goal_pose = ((pos[0], pos[1], block_size[2] / 2), rot)
            self.goals.append(
                ([pick_block], target_matrix, [goal_pose],
                 True, False, 'zone',
                 (block_pts, [(target_zone_pose, zone_size)]), 1)
            )
            self.lang_goals.append(self.lang_template.format(pick_color=block_colors[0],
                                                             place_color=zone_colors[0],
                                                             place=target_objs[target_idx]))
        else:
            raise ValueError("Unknown target object: {}".format(target_objs[target_idx]))

    @staticmethod
    def get_colors() -> Union[List[str], Set[str]]:
        return set(utils.TRAIN_COLORS + utils.EVAL_COLORS)


class PickAndPlacePrimitiveWithSize(Task):
    """
    Pick-and-place primitive for the LLM planner.
    This primitive is trained with all the colors, following the setting of `Inner Monologue`.
    In addition, this primitive is trained to discriminate two sizes: bigger and smaller.
    Pick up the [bigger/smaller] [block1] and place it on the [bigger/smaller] [block2/bowl/zone].
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = ("pick up the {pick_size} {pick_color} block and "
                              "place it on the {place_size} {place_color} block.")
        self.task_completed_desc = "done placing blocks."

    def reset(self, env):
        super().reset(env)
        all_color_names = self.get_colors()

        n_blocks = np.random.randint(2, 4)
        n_bowls = np.random.randint(2, 4)
        n_bigger_blocks = np.random.randint(2, n_blocks + 1)
        bigger_block_indexes = np.random.choice(list(range(n_blocks + n_bigger_blocks)),
                                                size=n_bigger_blocks,
                                                replace=False, )

        block_colors = random.sample(all_color_names, n_blocks)
        bowl_colors = random.sample(all_color_names, n_bowls)

        # Add bigger block and bowl colors, their colors are the same as smaller objects' colors,
        # bigger blocks and bowls are at the beginning part
        bigger_block_colors = block_colors[:n_bigger_blocks]
        for i, idx in enumerate(bigger_block_indexes):
            block_colors.insert(idx, bigger_block_colors[i])

        block_util_colors = [utils.COLORS[cn] for cn in block_colors]
        bowl_util_colors = [utils.COLORS[cn] for cn in bowl_colors]

        self.scene_description = (f"On the table, there are {n_blocks + n_bigger_blocks} blocks. "
                                  f"Their colors are {block_colors}. "
                                  f"There are {n_bowls} bowls. Their colors are {bowl_colors}. ")

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_util_colors[i] + [1])

        # Add blocks.
        block_size, bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        block_urdf, bigger_block_urdf = 'stacking/block.urdf', 'stacking/bigger_block.urdf'
        pick_block = None
        pick_obj_pose, place_obj_pose = None, None
        pick_block_idx = np.random.randint(0, n_blocks + n_bigger_blocks)
        place_block_idx = np.random.randint(0, n_blocks + n_bigger_blocks)
        while place_block_idx == pick_block_idx:
            place_block_idx = np.random.randint(0, n_blocks + n_bigger_blocks)

        for i in range(n_blocks + n_bigger_blocks):
            if i in bigger_block_indexes:
                block_pose = self.get_random_pose(env, bigger_block_size)
                block_id = env.add_object(bigger_block_urdf, block_pose)
            else:
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_util_colors[i] + [1])

            if i == pick_block_idx:
                pick_obj_pose = block_pose
                pick_block = (block_id, (np.pi / 2, None))
            elif i == place_block_idx:
                place_obj_pose = block_pose

        pick_size_desc = "bigger" if pick_block_idx in bigger_block_indexes else "smaller"
        place_block_size_desc = "bigger" if place_block_idx in bigger_block_indexes else "smaller"

        target_pos = (
            (place_obj_pose[0][0],
             place_obj_pose[0][1],
             place_obj_pose[0][2] * 2 + pick_obj_pose[0][2]),  # note the height is half of size[2]
            place_obj_pose[1]
        )
        # print("target_pos:", target_pos)
        # print(f"height, {place_obj_pose[0][2]}, {pick_obj_pose[0][2]}")
        self.goals.append(
            ([pick_block], np.eye(1), [target_pos], False, True, 'pose', None, 1)
        )
        self.lang_goals.append(
            self.lang_template.format(pick_size=pick_size_desc,
                                      pick_color=block_colors[pick_block_idx],
                                      place_size=place_block_size_desc,
                                      place_color=block_colors[place_block_idx], )
        )

    @staticmethod
    def get_colors() -> Union[List[str], Set[str]]:
        return set(utils.TRAIN_COLORS + utils.EVAL_COLORS)


class PickAndPlacePrimitiveWithRelativePosition(Task):
    """
    Pick-and-place primitive for the LLM planner.
    This primitive is trained with all the colors, following the setting of `Inner Monologue`.
    In addition, this primitive is trained to discriminate the relative position: top, down, left, right.
    Pick up the [block1] and place it on the [top/bottom/left/right of] [block2/bowl/zone].
    If the block2's rotation is not horizontal or vertical, the robot should change its rotation first.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = ("pick up the {pick_color} block and "
                              "place it on the {place_relative_pos} of {place_color} block")
        self.task_completed_desc = "done placing blocks."

    def reset(self, env):
        super().reset(env)
        all_color_names = self.get_colors()

        n_blocks = np.random.randint(2, 4)
        n_zones = np.random.randint(1, 3)
        n_bowls = np.random.randint(2, 4)
        rel_pos_list = ["top", "bottom", "left", "right"]
        rel_pos = random.choice(rel_pos_list)
        # n_blocks = 4
        # n_zones = 3
        # n_bowls = 4

        block_colors = random.sample(all_color_names, n_blocks)
        block_util_colors = [utils.COLORS[cn] for cn in block_colors]

        self.scene_description = f"On the table, there are {n_blocks} blocks. " \
                                 f"Their colors are {block_colors}. "

        # Add blocks.
        blocks = []
        block_pts = {}
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        pick_obj_pose, place_obj_pose = None, None
        if n_blocks < 2:
            raise ValueError("Too few blocks! Please increase the number of blocks!")
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_util_colors[i] + [1])
            if i == 0:
                pick_obj_pose = block_pose
            elif i == 1:
                block_pts[block_id] = self.get_box_object_points(block_id)
                place_obj_pose = block_pose
            blocks.append((block_id, (np.pi / 2, None)))

        theta = 0
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        place_obj_pose = (np.asarray(place_obj_pose[0]) + np.random.uniform(0.01, 0.02), rot)

        # Target pose
        # TODO: check whether the pos is correct.
        if rel_pos == "left":
            target_pos = (
                place_obj_pose[0][0],
                place_obj_pose[0][1] - block_size[0] * 4,
                place_obj_pose[0][2]
            )
        elif rel_pos == "right":
            target_pos = (
                place_obj_pose[0][0],
                place_obj_pose[0][1] + block_size[0] * 4,
                place_obj_pose[0][2]
            )
        elif rel_pos == "top":
            target_pos = (
                place_obj_pose[0][0] - block_size[1] * 4,
                place_obj_pose[0][1],
                place_obj_pose[0][2]
            )
        elif rel_pos == "bottom":
            target_pos = (
                place_obj_pose[0][0] + block_size[1] * 4,
                place_obj_pose[0][1],
                place_obj_pose[0][2]
            )
        else:
            raise ValueError("The value of rel_pos is wrong. Please check!")

        src = [blocks[1], blocks[0]]
        tgt = [place_obj_pose, (target_pos, place_obj_pose[1])]
        self.goals.append((src, np.eye(2),
                           tgt, False, False, 'pose', None, 1))
        # self.goals.append((blocks[:1], np.eye(1),
        #                    [(target_pos, place_obj_pose[1])], False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(
            pick_color=block_colors[0],
            place_relative_pos=rel_pos,
            place_color=block_colors[1],
        ))

    def get_colors(self) -> Union[List[str], Set[str]]:
        return set(utils.TRAIN_COLORS + utils.EVAL_COLORS)


class PickAndPlacePrimitiveWithAbsolutePosition(Task):
    """
    Pick-and-place primitive for the LLM planner.
    This primitive is trained with all the colors, following the setting of `Inner Monologue`.
    In addition, this primitive is trained to discriminate the absolute position: center of the table.
    Pick up the [block1] on the [pick_area] and place it on the [place_area].
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = ("pick up the {pick_color} block "
                              "on the {pick_area} and place it on the {place_area}.")
        self.task_completed_desc = "done placing blocks."

    def reset(self, env):
        super().reset(env)
        all_color_names = self.get_colors()

        n_blocks = np.random.randint(2, 4)
        n_bowls = np.random.randint(2, 4)

        block_colors = random.sample(all_color_names, n_blocks)
        bowl_colors = random.sample(all_color_names, n_bowls)
        block_util_colors = [utils.COLORS[cn] for cn in block_colors]
        bowl_util_colors = [utils.COLORS[cn] for cn in bowl_colors]

        block_size = (0.04, 0.04, 0.04)
        bowl_size = (0.12, 0.12, 0)
        height = block_size[2] / 2

        # Add bowls.
        bowl_urdf = 'bowl/bowl.urdf'
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_util_colors[i] + [1])

        # Add blocks.
        block_urdf = 'stacking/block.urdf'
        first_block, first_block_pose = None, None
        block_pts = {}
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_util_colors[i] + [1])
            if i == 0:
                block_pts[block_id] = self.get_box_object_points(block_id)
                first_block = (block_id, (np.pi / 2, None))
                first_block_pose = block_pose

        # Calculate which area the first block is on.
        pick_area = None
        pos, rot = first_block_pose
        for area_k, area_v in self.area_boundary.items():
            if (
                    (area_v["x_start"] <= pos[0] <= area_v["x_end"])
                    and
                    (area_v["y_start"] <= pos[1] <= area_v["y_end"])
            ):
                pick_area = area_k
                break

        area_candidates = [
            "center", "top left area", "top right area", "bottom left area", "bottom right area"
        ]
        area_candidates.remove(pick_area)
        place_area = random.choice(area_candidates)
        place_boundary = self.area_boundary[place_area]
        x_start, x_end = place_boundary["x_start"], place_boundary["x_end"]
        y_start, y_end = place_boundary["y_start"], place_boundary["y_end"]
        # place_pose = [
        #     (np.random.uniform(x_start, x_end), np.random.uniform(y_start, y_end), height),
        #     rot
        # ]

        place_pose = [
            (x_start + (x_end - x_start) / 2, y_start + (y_end - y_start) / 2, height),
            rot
        ]
        if (x_end - x_start == 0) and (y_end - y_start == 0):  # center
            place_area_size = (block_size[0] * 2, block_size[1] * 2, 0)
        else:
            place_area_size = (x_end - x_start, y_end - y_start, 0)

        self.goals.append(
            ([first_block], np.eye(1), [place_pose],
             True, False, 'zone',
             (block_pts, [(place_pose, place_area_size)]), 1)
        )
        self.lang_goals.append(self.lang_template.format(pick_color=block_colors[0],
                                                         pick_area=pick_area,
                                                         place_area=place_area))

        self.scene_description = (f"On the table, there are {n_blocks} blocks. "
                                  f"Their colors are {block_colors}. "
                                  f"There are {n_bowls} bowls. Their colors are {bowl_colors}. ")

    @staticmethod
    def get_colors() -> Union[List[str], Set[str]]:
        return set(utils.TRAIN_COLORS + utils.EVAL_COLORS)
