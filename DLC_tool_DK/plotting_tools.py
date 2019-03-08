"""
DeepLabCut2.0 Toolbox
D Kim, donniek@bcm.edu
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import time

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.plotting import get_cmap
# already configured for cv2
from deeplabcut.utils.video_processor import VideoProcessorCV as vp

# for ipython purpose
import pylab as pl
from IPython import display

def get_frame(path_to_video, frame_num):
    cap = cv2.VideoCapture(path_to_video)
    cap.open(path_to_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    _, img = cap.read()
    cap.release()
    return img


def df_part_generator(df_original, scorer, parts):

    df_part = df_original[scorer][parts]
    nframes = df_part.shape[0]
    df_part_likelihood = np.empty((len(parts), nframes))
    df_part_x = np.empty((len(parts), nframes))
    df_part_y = np.empty((len(parts), nframes))
    for bpindex, bp in enumerate(parts):
        df_part_likelihood[bpindex,:] = df_original[scorer][bp]['likelihood'].values
        df_part_x[bpindex, :] = df_original[scorer][bp]['x'].values
        df_part_y[bpindex, :] = df_original[scorer][bp]['y'].values
    
    return df_part, df_part_likelihood, df_part_x, df_part_y


def plot_over_frames(path_to_config, case, bodyparts2plot, starting, end, shuffle=1, save_fig=False, save_gif=False):
    #TODO change it such that it only plots one frame
    case_full_name = case + '_beh'

    config = auxiliaryfunctions.read_config(path_to_config)

    project_path = config['project_path']
    path_to_video = os.path.join(project_path, 'videos', case_full_name + '.avi')
    label_path = os.path.join(project_path, 'analysis', case_full_name )

    pcutoff = config['pcutoff']
    cropping = config['cropping']
    colormap = config['colormap']
    trainingsetindex = 0  # modify here as needed
    trainFraction = config['TrainingFraction'][trainingsetindex]
    DLCscorer = auxiliaryfunctions.GetScorerName(
        config, shuffle, trainFraction)

    # dotsize = config["dotsize"]
    dotsize = 5  # manually change here so that they stand out when plotting

    colormap = config["colormap"]
    alphavalue = config["alphavalue"]

    # obtain video
    clip = vp(fname=path_to_video)

    # label coordinate info
    df_label = pd.read_hdf(os.path.join(
        label_path, case_full_name + DLCscorer + '.h5'))

    df_part, df_part_likelihood, df_part_x, df_part_y = df_part_generator(df_original=df_label,
                                                                          scorer=DLCscorer,
                                                                          parts=bodyparts2plot)
    colors = get_cmap(len(bodyparts2plot), name=colormap)
    #colors_dict = dict(zip(pupil_parts, [colors(i) for i in range(colors.N)]))

    if config['cropping']:
        [x1, x2, y1, y2] = config['x1'], config['x2'], config['y1'], config['y2']
        ny, nx = y2-y1, x2-x1
    else:
        [x1, x2, y1, y2] = 0, clip.height(), 0, clip.width()
        ny, nx = clip.height(), clip.width()

    for index in range(starting, end):
        plt.axis('off')

        image = get_frame(path_to_video, frame_num=index)
        if cropping:
            image = image[y1:y2, x1:x2]
        else:
            pass
        fig = plt.figure(frameon=False, figsize=(12,8))
        # fig = plt.figure(frameon=False, figsize=(nx * 1. / 100, ny * 1. / 100))
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)
        plt.imshow(image)

        for bpindex, bp in enumerate(bodyparts2plot):
            if df_part_likelihood[bpindex, index] > pcutoff:
                plt.scatter(
                    df_part_x[bpindex, index],
                    df_part_y[bpindex, index],
                    s=dotsize**2,
                    color=colors(bpindex),
                    alpha=alphavalue
                )

        plt.xlim(0, nx)
        plt.ylim(0, ny)
        plt.axis('off')
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.gca().invert_yaxis()
        plt.title('frame num: ' + str(index), fontsize=30)

        sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(
            vmin=-0.5, vmax=len(bodyparts2plot)-0.5))
        sm._A = []
        cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
        cbar.set_ticklabels(bodyparts2plot)
        cbar.ax.tick_params(labelsize=20)

        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(1.0)

    if save_fig:
        plt.tight_layout()
        plt.savefig(os.path.join(label_path,'frame_' + str(starting) + '.png'))

    if save_gif:
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close('all')
        return image

    plt.close('all')
    return fig
    


def fit_pupil_over_frames(df_all_part, scorer, path_to_video, bodyparts2plot, starting, end, save_as_movie=False):

    df_part, df_part_likelihood, df_part_x, df_part_y = df_part_generator(df_original,
                                                                          scorer,
                                                                          bodyparts2plot)
    colors = get_cmap(len(bodyparts2plot), name=colormap)

    if config['cropping']:
        [x1, x2, y1, y2] = config['x1'], config['x2'], config['y1'], config['y2']
        ny, nx = y2-y1, x2-x1
    else:
        [x1, x2, y1, y2] = 0, clip.height(), 0, clip.width()
        ny, nx = clip.height(), clip.width()

    for index in range(starting, end):

        frame = get_frame(path_to_video, frame_num=index)
        plt.figure(frameon=False, figsize=(12, 6))
#         plt.figure(frameon=False, figsize=(nx * 1. / 100, ny * 1. / 100))
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)

        pupil_coords = []
        for bpindex, bp in enumerate(bodyparts2plot):
            if df_part_likelihood[bpindex, index] > pcutoff:
                plt.scatter(
                    df_part_x[bpindex, index]-x1,
                    df_part_y[bpindex, index]-y1,
                    s=dotsize**2,
                    color=colors(bpindex),
                    alpha=alphavalue
                )
                if bp in pupil_parts:
                    pupil_coords.append(
                        [df_part_x[bpindex, index], df_part_y[bpindex, index]])

        if len(pupil_coords) >= 3:
            pupil_coords = np.array(pupil_coords).reshape(-1, 1, 2).astype(int)
            (x, y), radius = cv2.minEnclosingCircle(pupil_coords)
            center = (int(x), int(y))
            radius = int(radius)
            frame = cv2.circle(frame, center, radius, (0, 255, 0), thickness=1)

        if cropping:
            frame = frame[y1:y2, x1:x2]
        else:
            pass

        plt.xlim(0, nx)
        plt.ylim(0, ny)
        plt.axis('off')
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.gca().invert_yaxis()
        plt.title('frame num: ' + str(index), fontsize=16)

        plt.imshow(frame)

        sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(
            vmin=-0.5, vmax=len(bodyparts2plot)-0.5))
        sm._A = []
        cbar = plt.colorbar(sm, ticks=range(len(bodyparts2plot)))
        cbar.set_ticklabels(bodyparts2plot)

        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(0.1)

    if save_as_movie:
        plt.tight_layout()
        print("image name is {}".format(label_path,'frame_' + str(starting) + '.png'))
        plt.savefig(os.path.join(label_path,'frame_' + str(starting) + '.png'))

    plt.close('all')
