import random
from typing import List

import numpy as np
import pybullet as p

from cliport.tasks.color_reasoning import COLOR_CATEGORY_NAMES
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveBlocks(Task):
    def __init__(self):
        super().__init__()

        self.objs = []
        self.smaller_block_size, self.bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        self.smaller_block_urdf, self.bigger_block_urdf = ("stacking/block.urdf",
                                                           "stacking/bigger_block.urdf")
        self.all_color_names = self.get_colors()
        self.all_block_colors = None
        self.areas = ["top left area", "top right area", "bottom left area", "bottom right area"]
        self.src_abs_area, self.tgt_abs_area = None, None

        self.n_block_distractors, self.n_zone_distractors = None, None
        self.min_move, self.max_move = None, None

    def reset(self, env):
        super().reset(env)
        # self.print_debug_info = False
        # self.input_manipulate_order = True
        self.consider_z_in_match = False

        self.min_move, self.max_move = 3, 5
        random.shuffle(self.all_color_names)
        self.all_block_colors = [utils.COLORS[cn] for cn in self.all_color_names]
        self.n_block_distractors = (np.random.randint(0, 5)
                                    if self.task_difficulty_level == "easy"
                                    else np.random.randint(5, 10))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS

    @staticmethod
    def classify_color(color: str) -> List[str]:
        categories = []
        for category, colors in COLOR_CATEGORY_NAMES.items():
            if color in colors:
                categories.append(category)
        return categories

    def is_pose_in_area(self, pose, area):
        pos, _ = pose
        if (
                (
                        self.area_boundary[area]["x_start"]
                        <= pos[0] <=
                        self.area_boundary[area]["x_end"]
                )
                and
                (
                        self.area_boundary[area]["y_start"]
                        <= pos[1] <=
                        self.area_boundary[area]["y_end"]
                )
        ):
            return True
        else:
            return False

    def random_pose_in_area(self, env, area, obj_size):
        pose = self.get_random_pose(env, obj_size)
        while not self.is_pose_in_area(pose, area):
            pose = self.get_random_pose(env, obj_size)
        return pose

    def get_selected_colors(self, n_move, selectable_colors) -> List[str]:
        """
        return the color list of blocks that are to be moved.
        n_move: number of blocks to be moved
        selectable_colors: list of selectable colors
        """
        if self.task_difficulty_level != "hard":
            selected_colors = np.random.choice(
                selectable_colors,
                size=n_move,
                replace=False
            ).tolist()
        else:  # allow duplicate colors
            selected_colors = np.random.choice(
                selectable_colors,
                size=n_move,
                replace=True,
            ).tolist()
        return selected_colors

    def get_target_poses_and_area_info(
            self,
            env,
            n_move: int,
    ):
        tgt_poses = []
        for _ in range(n_move):
            tgt_pose = self.random_pose_in_area(env, self.tgt_abs_area, 2 * self.bigger_block_size)
            tgt_poses.append(tgt_pose)

        theta = 0
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        tgt_area_pose = (
            (
                self.area_boundary[self.tgt_abs_area]["x_start"] + self.x_length / 4,
                self.area_boundary[self.tgt_abs_area]["y_start"] + self.y_length / 4,
                0
            ),
            rot
        )
        tgt_area_size = (self.x_length / 2 * 1.2, self.y_length / 2 * 1.2, 0)

        return tgt_poses, tgt_area_pose, tgt_area_size


class MoveBlocksBetweenAbsolutePositions(MoveBlocks):
    """Move all the blocks on the [X] area to [Y] area.
    X and Y are sampled from [top left, top right, bottom left, bottom right]
    Example: ``Move all the blocks on the top left area to bottom right area".
    """

    def __init__(self):
        super().__init__()
        self.src_abs_pos, self.tgt_abs_pos = None, None
        self.lang_template = "Move all the blocks on the {src_abs_pos} to the {tgt_abs_pos}."
        self.task_completed_desc = "Done moving all the blocks."

    def reset(self, env):
        super().reset(env)

        block_size, block_urdf = self.smaller_block_size, self.smaller_block_urdf
        self.min_move, self.max_move = (3, 6) if self.task_difficulty_level == "easy" else (5, 9)
        self.src_abs_area, self.tgt_abs_area = np.random.choice(self.areas, size=2, replace=False)
        n_move = np.random.randint(self.min_move, self.max_move)

        selectable_colors = self.all_color_names
        n_move = min(n_move, len(selectable_colors))
        self.max_steps = n_move + 7

        selected_colors = self.get_selected_colors(n_move, selectable_colors)
        selected_colors_palettes = [utils.COLORS[c] for c in selected_colors]

        # Add blocks that are to be moved.
        blocks_pts = {}
        selected_blocks = []
        for i in range(n_move):
            block_pose = self.get_random_pose(env, block_size)
            while not self.is_pose_in_area(block_pose, self.src_abs_area):
                block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(
                block_id,
                -1,
                rgbaColor=selected_colors_palettes[i] + [1]
            )
            selected_blocks.append((block_id, (np.pi / 2, None)))
            blocks_pts[block_id] = self.get_box_object_points(block_id)

        distractors_colors = []
        if self.n_block_distractors:
            distractors_colors = [c for c in self.all_color_names if c not in selected_colors]
            if len(distractors_colors) < self.n_block_distractors:
                distractors_colors += np.random.choice(
                    distractors_colors,
                    size=self.n_block_distractors - len(distractors_colors),
                    replace=True
                ).tolist()
            distractors_colors_palettes = [utils.COLORS[c] for c in distractors_colors]
            for i in range(self.n_block_distractors):
                block_pose = self.get_random_pose(env, block_size)
                while self.is_pose_in_area(block_pose, self.src_abs_area):
                    block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose)
                p.changeVisualShape(
                    block_id,
                    -1,
                    rgbaColor=distractors_colors_palettes[i] + [1]
                )

        selected_blocks_pts = {k[0]: blocks_pts[k[0]] for k in selected_blocks}

        tgt_poses, tgt_area_pose, tgt_area_size = self.get_target_poses_and_area_info(
            env, n_move,
        )

        self.goals.append(
            (
                selected_blocks,
                np.eye(n_move),
                tgt_poses,
                False, True, 'zone',
                (selected_blocks_pts, [(tgt_area_pose, tgt_area_size)]),
                1,
            )
        )
        self.lang_goals.append(
            self.lang_template.format(src_abs_pos=self.src_abs_area,
                                      tgt_abs_pos=self.tgt_abs_area)
        )

        self.scene_description = (f"On the table, "
                                  f"there are {n_move + self.n_block_distractors} blocks. "
                                  f"Their colors are "
                                  f"{', '.join(selected_colors + distractors_colors)}. ")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class MoveBlocksBetweenAbsolutePositionsByColor(MoveBlocks):
    """
    Task instruction:
    "Move all the {select_color_category} blocks
    on the {source_abs_pos} area to the {target_abs_pos} area."
    Note each block's color is unique thus does not require reference capability.
    """

    def __init__(self):
        super().__init__()

        self.lang_template = ("Move all the blocks of {color_categories} "
                              "on the {src_abs_pos} to the {tgt_abs_pos}.")
        self.task_completed_desc = "Done moving blocks."

    def reset(self, env):
        super().reset(env)

        block_size, block_urdf = self.smaller_block_size, self.smaller_block_urdf
        self.min_move, self.max_move = (3, 6) if self.task_difficulty_level != "hard" else (5, 9)
        self.src_abs_area, self.tgt_abs_area = np.random.choice(self.areas, size=2, replace=False)

        n_color_categories = np.random.randint(1, len(COLOR_CATEGORY_NAMES.keys()) + 1)
        select_color_categories = np.random.choice(
            list(COLOR_CATEGORY_NAMES.keys()),
            size=n_color_categories,
            replace=False,
        )
        selectable_colors = []
        for i in range(len(select_color_categories)):
            selectable_colors += COLOR_CATEGORY_NAMES[select_color_categories[i]]
        selectable_colors = list(dict.fromkeys(selectable_colors))  # remove duplicate

        n_move = np.random.randint(self.min_move, self.max_move)
        n_move = min(n_move, len(selectable_colors))
        self.max_steps = n_move + 7

        selected_colors = self.get_selected_colors(n_move, selectable_colors)
        selected_color_palettes = [utils.COLORS[c] for c in selected_colors]

        # Add blocks that are to be moved.
        blocks_pts = {}
        selected_blocks = []
        for i in range(n_move):
            block_pose = self.get_random_pose(env, block_size)
            while not self.is_pose_in_area(block_pose, self.src_abs_area):
                block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(
                block_id,
                -1,
                rgbaColor=selected_color_palettes[i] + [1]
            )
            selected_blocks.append((block_id, (np.pi / 2, None)))
            blocks_pts[block_id] = self.get_box_object_points(block_id)

        distractors_colors = []
        if self.n_block_distractors:
            distractors_colors = [c for c in self.all_color_names if c not in selectable_colors]
            if distractors_colors:
                if len(distractors_colors) < self.n_block_distractors:
                    distractors_colors += np.random.choice(
                        distractors_colors,
                        size=self.n_block_distractors - len(distractors_colors),
                        replace=True
                    ).tolist()
                distractors_colors_palettes = [utils.COLORS[c] for c in distractors_colors]
                for i in range(self.n_block_distractors):
                    block_pose = self.get_random_pose(env, block_size)
                    block_id = env.add_object(block_urdf, block_pose)
                    p.changeVisualShape(
                        block_id,
                        -1,
                        rgbaColor=distractors_colors_palettes[i] + [1]
                    )

        selected_blocks_pts = {k[0]: blocks_pts[k[0]] for k in selected_blocks}

        tgt_poses, tgt_area_pose, tgt_area_size = self.get_target_poses_and_area_info(
            env, n_move,
        )

        self.goals.append(
            (
                selected_blocks,
                np.eye(n_move),
                tgt_poses,
                False, True, 'zone',
                (selected_blocks_pts, [(tgt_area_pose, tgt_area_size)]),
                1,
            )
        )
        self.lang_goals.append(
            self.lang_template.format(
                color_categories=f"{' colors and '.join(select_color_categories)} colors",
                src_abs_pos=self.src_abs_area,
                tgt_abs_pos=self.tgt_abs_area
            )
        )

        self.scene_description = (f"On the table, "
                                  f"there are {n_move + self.n_block_distractors} blocks. "
                                  f"Their colors are "
                                  f"{', '.join(selected_colors + distractors_colors)}. ")


class MoveBlocksBetweenAbsolutePositionsBySize(MoveBlocks):
    """
    Task instruction:
    "Move all the bigger/smaller blocks
    on the {source_abs_pos} area to the {target_abs_pos} area."
    Note each block's color is unique thus does not require reference capability.
    """

    def __init__(self):
        super().__init__()

        self.lang_template = ("Move all the {size_category} blocks "
                              "on the {src_abs_pos} to the {tgt_abs_pos}.")
        self.task_completed_desc = "Done moving blocks."

    def reset(self, env):
        super().reset(env)

        size_category = np.random.choice(["smaller", "bigger"])
        block_size, block_urdf = ((self.smaller_block_size, self.smaller_block_urdf)
                                  if size_category == "smaller"
                                  else (self.bigger_block_size, self.bigger_block_urdf))

        self.min_move, self.max_move = (3, 6) if self.task_difficulty_level != "hard" else (5, 9)
        self.src_abs_area, self.tgt_abs_area = np.random.choice(self.areas, size=2, replace=False)
        n_move = np.random.randint(self.min_move, self.max_move)

        selectable_colors = self.all_color_names
        n_move = min(n_move, len(selectable_colors))
        self.max_steps = n_move + 7

        selected_colors = self.get_selected_colors(n_move, selectable_colors)
        selected_colors_palettes = [utils.COLORS[c] for c in selected_colors]

        # Add blocks that are to be moved.
        blocks_pts = {}
        selected_blocks = []
        for i in range(n_move):
            block_pose = self.get_random_pose(env, block_size)
            while not self.is_pose_in_area(block_pose, self.src_abs_area):
                block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(
                block_id,
                -1,
                rgbaColor=selected_colors_palettes[i] + [1]
            )
            selected_blocks.append((block_id, (np.pi / 2, None)))
            blocks_pts[block_id] = self.get_box_object_points(block_id)

        distractors_colors = []
        if self.n_block_distractors:
            distractors_colors = self.all_color_names
            if len(distractors_colors) < self.n_block_distractors:
                distractors_colors += np.random.choice(
                    distractors_colors,
                    size=self.n_block_distractors - len(distractors_colors),
                    replace=True
                ).tolist()
            distractors_colors_palettes = [utils.COLORS[c] for c in distractors_colors]

            distractors_size_category = "bigger" if size_category == "smaller" else "smaller"
            distractor_block_size, distractor_block_urdf = (
                (self.smaller_block_size, self.smaller_block_urdf)
                if distractors_size_category == "smaller"
                else (self.bigger_block_size, self.bigger_block_urdf)
            )
            for i in range(self.n_block_distractors):
                block_pose = self.get_random_pose(env, distractor_block_size)
                block_id = env.add_object(distractor_block_urdf, block_pose)
                p.changeVisualShape(
                    block_id,
                    -1,
                    rgbaColor=distractors_colors_palettes[i] + [1]
                )

        selected_blocks_pts = {k[0]: blocks_pts[k[0]] for k in selected_blocks}

        tgt_poses, tgt_area_pose, tgt_area_size = self.get_target_poses_and_area_info(
            env, n_move,
        )

        self.goals.append(
            (
                selected_blocks,
                np.eye(n_move),
                tgt_poses,
                False, True, 'zone',
                (selected_blocks_pts, [(tgt_area_pose, tgt_area_size)]),
                1,
            )
        )
        self.lang_goals.append(
            self.lang_template.format(size_category=size_category,
                                      src_abs_pos=self.src_abs_area,
                                      tgt_abs_pos=self.tgt_abs_area)
        )

        self.scene_description = (f"On the table, "
                                  f"there are {n_move + self.n_block_distractors} blocks. "
                                  f"Their colors are "
                                  f"{', '.join(selected_colors + distractors_colors)}. "
                                  f"The sizes of blocks may be different. "
                                  f"Some blocks are 'bigger' and some blocks are 'smaller'.")


class MoveBlocksBetweenAbsolutePositionsBySizeAndColor(MoveBlocks):
    """
    Task instruction:
    "Move all the bigger/smaller blocks of primary/secondary/warm/cool color
    on the {source_abs_pos} area to the {target_abs_pos} area."
    We add more distractors to make this task more difficult.
    """

    def __init__(self):
        super().__init__()

        self.lang_template = ("Move all the {size_category} blocks of {color_categories} "
                              "on the {src_abs_pos} to the {tgt_abs_pos}.")
        self.task_completed_desc = "Done moving blocks."

    def reset(self, env):
        super().reset(env)

        size_category = np.random.choice(["smaller", "bigger"])
        block_size, block_urdf = ((self.smaller_block_size, self.smaller_block_urdf)
                                  if size_category == "smaller"
                                  else (self.bigger_block_size, self.bigger_block_urdf))

        self.min_move, self.max_move = (3, 6) if self.task_difficulty_level != "hard" else (5, 9)
        self.src_abs_area, self.tgt_abs_area = np.random.choice(self.areas, size=2, replace=False)

        n_color_categories = np.random.randint(1, len(COLOR_CATEGORY_NAMES.keys()) + 1)
        select_color_categories = np.random.choice(
            list(COLOR_CATEGORY_NAMES.keys()),
            size=n_color_categories,
            replace=False,
        )
        selectable_colors = []
        for i in range(len(select_color_categories)):
            selectable_colors += COLOR_CATEGORY_NAMES[select_color_categories[i]]
        selectable_colors = list(dict.fromkeys(selectable_colors))  # remove duplicate

        n_move = np.random.randint(self.min_move, self.max_move)
        n_move = min(n_move, len(selectable_colors))
        self.max_steps = n_move + 7

        selected_colors = self.get_selected_colors(n_move, selectable_colors)
        selected_color_palettes = [utils.COLORS[c] for c in selected_colors]

        # Add blocks that are to be moved.
        blocks_pts = {}
        selected_blocks = []
        for i in range(n_move):
            block_pose = self.get_random_pose(env, block_size)
            while not self.is_pose_in_area(block_pose, self.src_abs_area):
                block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(
                block_id,
                -1,
                rgbaColor=selected_color_palettes[i] + [1]
            )
            selected_blocks.append((block_id, (np.pi / 2, None)))
            blocks_pts[block_id] = self.get_box_object_points(block_id)

        distractors_colors = []
        if self.n_block_distractors:
            distractors_size_category = np.random.choice(["smaller", "bigger"])
            distractors_colors = ([c for c in self.all_color_names if c not in selectable_colors]
                                  if distractors_size_category == size_category
                                  else self.all_color_names)
            if distractors_colors:
                if len(distractors_colors) < self.n_block_distractors:
                    distractors_colors += np.random.choice(
                        distractors_colors,
                        size=self.n_block_distractors - len(distractors_colors),
                        replace=True
                    ).tolist()
                distractors_colors_palettes = [utils.COLORS[c] for c in distractors_colors]
                for i in range(self.n_block_distractors):
                    block_size, block_urdf = (
                        (self.smaller_block_size, self.smaller_block_urdf)
                        if distractors_size_category == "smaller"
                        else (self.bigger_block_size, self.bigger_block_urdf)
                    )
                    block_pose = self.get_random_pose(env, block_size)
                    block_id = env.add_object(block_urdf, block_pose)
                    p.changeVisualShape(
                        block_id,
                        -1,
                        rgbaColor=distractors_colors_palettes[i] + [1]
                    )

        selected_blocks_pts = {k[0]: blocks_pts[k[0]] for k in selected_blocks}

        tgt_poses, tgt_area_pose, tgt_area_size = self.get_target_poses_and_area_info(
            env, n_move,
        )

        self.goals.append(
            (
                selected_blocks,
                np.eye(n_move),
                tgt_poses,
                False, True, 'zone',
                (selected_blocks_pts, [(tgt_area_pose, tgt_area_size)]),
                1,
            )
        )
        self.lang_goals.append(
            self.lang_template.format(
                size_category=size_category,
                color_categories=f"{' colors and '.join(select_color_categories)} colors",
                src_abs_pos=self.src_abs_area,
                tgt_abs_pos=self.tgt_abs_area
            )
        )

        self.scene_description = (f"On the table, "
                                  f"there are {n_move + self.n_block_distractors} blocks. "
                                  f"Their colors are "
                                  f"{', '.join(selected_colors + distractors_colors)}. ")
