import numpy as np
import cv2
import copy
import pickle
import os
import math
import click
import pandas as pd

@click.group()
@click.version_option()
def cli():
    pass


def transform(point, base):
    point_t = (point - base) * 10 + 400
    return (int(point_t[0]), int(point_t[1]))


def draw_trajectory(canvas, trajectory, attribute_line, base, traj_info=[]):
    """Visualize trajectory.

    loc: red [0, 0, 255]
    Ground truth: green [0, 255, 0]
    Prediction:  blue [255, 0, 0]
    Proposal: brone 
    Others: 

    Args:
        canvas: A canvas.
        trajectory: locations [N, 2].
        attribute_line: A dict indicates color, size and thick.
        traj_info: A dict of shown text.
        base: transform center point

    Returns:
        canvas
    """
    for point in trajectory:
        cv2.circle(canvas, transform(point, base), attribute_line['size'],
                   attribute_line['color'], attribute_line['thick'])

    # draw start point
    cv2.circle(canvas, transform(trajectory[0], base), attribute_line['size'],
               attribute_line['color'], -1)

    # draw text information
    start_x = 40
    start_y = 40
    step = 60
    for txt in traj_info:
        cv2.putText(canvas, '{} : {}'.format(txt, traj_info[txt]), (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    attribute_line['color'], 2)
        start_y += step

    return canvas


@cli.command()
@click.option('--data_path', default='data/full_data/gt.pkl')
@click.option('--save_path', default='vis_results')
def vis_exported_data(data_path, save_path):
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    root_path = os.path.join(*data_path.split('/')[:-1])
    file_name = 'offline_video'
    fps = 16
    img_width = 800
    img_height = 800
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    # read excel
    trajectory_history_df = pd.read_excel(data_path, index_col=None, sheet_name='Sheet1')
    loc_x = np.array(trajectory_history_df["x"])
    loc_y = np.array(trajectory_history_df["y"])
    v_x = np.array(trajectory_history_df["vx"])
    v_y = np.array(trajectory_history_df["vy"])
    all_yaw = np.array(trajectory_history_df["yaw"])
    all_veh_ids = np.array(trajectory_history_df["vehicle_id"])

    # visualization
    max_vis_length = 1000
    veh_ids = np.unique(all_veh_ids)
    for veh_id in veh_ids:
        video_writer = cv2.VideoWriter('{}/{}.mp4'.format(root_path, veh_id), fourcc, fps, (img_height, img_width))
        print('===> write to: ', '{}/{}.mp4'.format(root_path, veh_id))
        indexs = np.where(all_veh_ids == veh_id)[0][::3]
        trajectories = np.concatenate([loc_x[indexs, np.newaxis], loc_y[indexs, np.newaxis]], axis=-1)
        velocities = np.concatenate([v_x[indexs, np.newaxis], v_y[indexs, np.newaxis]], axis=-1)
        v_all = 3.6 * np.linalg.norm(velocities, axis=-1)
        this_yaw = all_yaw[indexs]
        # import pdb;pdb.set_trace()
        for frame_id in range(len(trajectories)):
            if frame_id < max_vis_length:
                trajectory = trajectories[:frame_id+1]
            else:
                trajectory = trajectories[frame_id-max_vis_length:frame_id+1]
            # transform
            current_yaw = this_yaw[frame_id]
            R_matrix = np.array([[np.cos(current_yaw), -np.sin(current_yaw)],[np.sin(current_yaw), np.cos(current_yaw)]])
            current_t = trajectory[-1]
            image_90 = math.pi / 2
            rotate_90 = np.array([[np.cos(image_90), -np.sin(image_90)],[np.sin(image_90), np.cos(image_90)]])
            trajectory = np.dot(R_matrix.T, (trajectory - current_t).T).T
            trajectory = np.dot(rotate_90.T, trajectory.T).T
            # trajectory[:, 0] = -trajectory[:, 0]

            attribute_line = {'color': (0, 0, 255), 'size': 4, 'thick': 1}
            canvas = np.ones([img_height, img_width, 3], dtype=np.uint8) * 255
            base = trajectory[-1]
            for point in trajectory:
                cv2.circle(canvas, transform(point, base), attribute_line['size'],
                        attribute_line['color'], attribute_line['thick'])

            # draw current point
            cv2.circle(canvas, transform(trajectory[-1], base), 2 * attribute_line['size'],
                    attribute_line['color'], -1)

            # draw text information
            start_x = 40
            start_y = 40
            step = 60
            cv2.putText(
                canvas, 'frame id: {}'.format(
                    frame_id), (start_x, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, attribute_line['color'], 2)
            start_y += step
            cv2.putText(
                canvas, 'vx: {:.2f} m/s, vy: {:.2f} m/s, v: {:.2f} km/h'.format(
                    velocities[frame_id, 0],
                    velocities[frame_id, 1], v_all[frame_id]), (start_x, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, attribute_line['color'], 2)

            # save image
            video_writer.write(canvas)
        video_writer.release()


if __name__ == '__main__':
    cli()
