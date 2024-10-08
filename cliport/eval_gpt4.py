"""Ravens main training script."""

import base64
import json
import logging
import os
from pathlib import Path
from time import sleep
from typing import Dict, List, Any, Union, Tuple

import hydra
import numpy as np
import openai
from matplotlib import pyplot as plt
from openai import RateLimitError, APIConnectionError, APIError

from cliport import agents
from cliport import dataset
from cliport import gpt_prompts
from cliport import tasks
from cliport.environments.environment import Environment
from cliport.utils import utils


# Function to encode the image
def encode_image(image_path):
    with Path(image_path).open(mode="rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_description_from_gpt(gpt_client, base64_image) -> str:
    prompt = [
        {
            "role": "system",
            "content": gpt_prompts.SYSTEM_PROMPT_FOR_REPORTER,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's the textual description of "
                            "the current observation as shown in the image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
            ],
        },
    ]
    while True:
        try:
            response = gpt_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt,
                max_tokens=500,
            )
            break
        except (RateLimitError, APIConnectionError, APIError) as e:
            print(f'OpenAI API got err {e}')
            print('Retrying after 10s.')
            sleep(10)
    return response.choices[0].message.content


def get_next_step_plan_from_gpt(
        gpt_client: openai.OpenAI,
        prompt_history: List[Dict[str, Any]],
        current_observation_path: Union[str, Path],
        describe_observation_as_text: bool = False,

) -> Union[str, Tuple[str, str]]:
    base64_image = encode_image(current_observation_path)
    if not describe_observation_as_text:
        prompt = prompt_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the plan for the next step based on "
                                "the current observation as shown in the image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                ],
            }
        ]
        while True:
            try:
                response = gpt_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=prompt,
                    max_tokens=300,
                )
                break
            except (RateLimitError, APIConnectionError, APIError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)
        return response.choices[0].message.content
    else:
        observation_description = get_image_description_from_gpt(gpt_client, base64_image)
        prompt = prompt_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Current observation is described "
                                f"as follows: {observation_description}. "
                                "What's the plan for the next step based on "
                                "the above observation description?"
                    },
                ],
            }
        ]
        while True:
            try:
                response = gpt_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=prompt,
                    max_tokens=300,
                )
                break
            except (RateLimitError, APIConnectionError, APIError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)
        return response.choices[0].message.content, observation_description


def get_current_observation(env, camera_view: str = "front", mode: str = "rgb_array"):
    if mode == "rgb_array" and camera_view == "front":
        return env.render(mode=mode)
    elif mode == "rgb_array" and camera_view == "top":
        color, _, _ = env.render_camera(env.agent_cams[3])
        return color
    elif mode == "rgb+depth" and camera_view == "front":
        color, depth, _ = env.render_camera(env.agent_cams[0])
    else:
        raise ValueError(f"mode {mode} and camera view {camera_view} not supported")


@hydra.main(config_path='./cfg', config_name='eval_gpt4')
def main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])
    gpt_client = openai.OpenAI(
        api_key=vcfg["openai_api_key"],
        organization=vcfg["openai_organization_id"],
        project=vcfg["openai_project_id"],
    )

    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    # Choose eval mode and task.
    mode = vcfg['mode']
    eval_task = vcfg['eval_task']
    if mode not in {'train', 'val', 'test'}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg['type']
    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(
            vcfg['data_dir'],
            tcfg,
            group=eval_task,
            mode=mode,
            n_demos=vcfg['n_demos'],
            augment=False
        )
    else:
        ds = dataset.RavensDataset(
            os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
            tcfg,
            n_demos=vcfg['n_demos'],
            augment=False
        )

    all_results = {}
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])

    # Save path for results.
    json_name = (f"multi-results-{mode}.json" if 'multi' in vcfg['model_path']
                 else f"results-{mode}-vlm.json")
    save_path = vcfg['save_path']
    save_image_path = f"{save_path}/save_images/"
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f'{name}-{json_name}')

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, 'r') as f:
            existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    primitive_ckpt_path = vcfg["primitive_ckpt_path"]

    results = []
    mean_reward = 0.0
    mean_binary_reward = 0.0

    # Run testing for each training run.
    for train_run in range(vcfg['n_repeats']):
        # Initialize agent.
        utils.set_seed(train_run, torch=True)
        agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

        # Load checkpoint
        logging.info(f"Loading primitive checkpoint: {primitive_ckpt_path}")
        agent.load(primitive_ckpt_path)

        record = vcfg['record']['save_video']
        n_demos = vcfg['n_demos']

        # Run testing and save total rewards with last transition info.
        for i in range(0, n_demos):
            logging.info(f'Test: {i + 1}/{n_demos}')
            episode, seed = ds.load(i)
            goal = episode[-1]
            total_reward = 0
            np.random.seed(seed)

            # set task
            if 'multi' in dataset_type:
                task_name = ds.get_curr_task()
                task = tasks.names[task_name]()
                print(f'Evaluating on {task_name}')
            else:
                task_name = vcfg['eval_task']
                task = tasks.names[task_name]()

            task.mode = mode
            env.seed(seed)
            env.set_task(
                task)  # SQ: I think this should be a key of interaction between env and task
            obs = env.reset()
            info = env.info  # SQ: Final language goal and env info with object poses, rotation, and colors

            # Start recording video (NOTE: super slow)
            if record:
                if not Path(vcfg['record']['save_video_path']).exists():
                    Path(vcfg['record']['save_video_path']).mkdir()
                logging.info("Start recording video ......")
                video_name = f"{task_name}{vcfg['note']}-camera_{vcfg['camera_view']}-{i + 1:06d}"
                if 'multi' in vcfg['model_task']:
                    video_name = f"{vcfg['model_task']}-{video_name}"
                env.start_rec(video_name)

            final_goal = info["lang_goal"]
            print(f'The final goal: {final_goal}')

            done = False
            n_steps = 0
            prompt_history = [
                {
                    "role": "system",
                    "content": gpt_prompts.SYSTEM_PROMPT_FOR_PLANNER,
                },
                {
                    "role": "user",
                    "content": f"The instruction is {final_goal}."
                               "Please plan for the robot to finish the instruction. "
                }
            ]

            write_plans = [f"Final goal: {final_goal}"]
            write_observation_descriptions = []
            binary_reward = 0
            while not done and n_steps < env.task.max_steps:
                if vcfg["image_input"] == "rgb+depth":
                    cur_img_color, cur_img_depth = get_current_observation(
                        env,
                        camera_view=vcfg['camera_view']
                    )
                elif vcfg["image_input"] == "rgb":
                    cur_img_color = get_current_observation(env, camera_view=vcfg['camera_view'])
                else:
                    raise ValueError(f"Invalid image_input: {vcfg['image_input']}")
                cur_img_path = (f"{save_image_path}/{task_name}{vcfg['note']}/"
                                f"camera_{vcfg['camera_view']}_demo_{i + 1}_step_{n_steps:02}.png")
                if not Path(cur_img_path).parent.exists():
                    Path(cur_img_path).parent.mkdir(parents=True, exist_ok=True)
                plt.imsave(cur_img_path, cur_img_color)

                if not vcfg["describe_observation_as_text"]:
                    print("Use GPT-4 as Planner!")
                    next_plan = get_next_step_plan_from_gpt(
                        gpt_client,
                        prompt_history,
                        cur_img_path,
                        describe_observation_as_text=False
                    )
                else:
                    print("Use GPT-4 as Planner and Reporter!")
                    next_plan, observation_description = get_next_step_plan_from_gpt(
                        gpt_client,
                        prompt_history,
                        cur_img_path,
                        describe_observation_as_text=True
                    )
                    write_observation_descriptions.append(f"Step {n_steps:02}: "
                                                          f"{observation_description}")

                logging.info(f"Next_plan is: {next_plan}")
                info['lang_goal'] = next_plan
                prompt_history.append(
                    {
                        "role": "assistant",
                        "content": next_plan,
                    }
                )
                write_plans.append(f"Step {n_steps:02}: {next_plan}")

                act = agent.act(obs, info, goal)  # SQ: goal is not used in the
                # implementation of act.
                lang_goal = info['lang_goal']
                obs, reward, done, info = env.step(act)
                n_steps += 1
                total_reward += reward
                print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
                if done:
                    binary_reward += 1
                    break

            # TODO: save the image after executing the last step

            if vcfg["write_plan_to_file"]:
                file_path = f"{save_image_path}/{task_name}{vcfg['note']}/"
                file_name = f"camera_{vcfg['camera_view']}_demo_{i + 1:02}.txt"
                plan_file = file_path + file_name
                with Path(plan_file).open('w', encoding="utf-8") as f:
                    f.writelines([f"{plan}\n" for plan in write_plans])
                    f.write("\n\nObservation Descriptions:\n")
                    if vcfg["describe_observation_as_text"]:
                        f.writelines([f"{obs}\n" for obs in write_observation_descriptions])

            results.append((total_reward, info, binary_reward))
            mean_reward = np.mean([r for r, _, _ in results])
            mean_binary_reward = np.mean([br for _, _, br in results])
            print(f'Mean reward: {mean_reward} | Task: {task_name} |')
            print(f'Mean binary reward: {mean_binary_reward} | Task: {task_name} |')

            # End recording video
            if record:
                env.end_rec()

        all_results[primitive_ckpt_path] = {
            'episodes': results,
            'mean_reward': mean_reward,
        }

    # Save results in a json file.
    if vcfg['save_results']:

        # Load existing results
        if os.path.exists(save_json):
            with open(save_json, 'r') as f:
                existing_results = json.load(f)
            existing_results.update(all_results)
            all_results = existing_results

        with open(save_json, 'w') as f:
            json.dump(all_results, f, indent=4)


if __name__ == '__main__':
    main()
