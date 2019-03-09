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
import imageio

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


def make_gif(case, start_frame_num, num_frame):
    save_dir = os.path.join(mother_dir, 'analysis', case+'_beh', case+'.gif')

    fig = plt.figure.Figure()
    plt_list = []

    for i in range(start_frame_num, start_frame_num + num_frame):
        plot = plot_over_frames(config_path,
                                case,
                                bodyparts2plot=all_parts,
                                starting=i,
                                end=i+1,
                                save_fig=False,
                                save_gif=True)
        plt_list.append(plot)
    imageio.mimsave(save_dir, plt_list, fps=1)


def bodyparts_info(config, case, bodyparts, trainingsetindex=0, shuffle=1):
    """
    Given bodyparts, return corresponding likelihood, x-coordinates, and y-coordinates
    """
    case_full_name = case + '_beh'

    project_path = config['project_path']
    label_path = os.path.join(project_path, 'analysis', case_full_name)

    trainFraction = config['TrainingFraction'][trainingsetindex]
    DLCscorer = auxiliaryfunctions.GetScorerName(
        config, shuffle, trainFraction)

    df_label = pd.read_hdf(os.path.join(
        label_path, case_full_name + DLCscorer + '.h5'))

    df_bodyparts = df_label[DLCscorer][bodyparts]
    nframes = df_bodyparts.shape[0]

    df_bodyparts_likelihood = np.empty((len(bodyparts), nframes))
    df_bodyparts_x = np.empty((len(bodyparts), nframes))
    df_bodyparts_y = np.empty((len(bodyparts), nframes))

    for bpindex, bp in enumerate(bodyparts):
        df_bodyparts_likelihood[bpindex,
                                :] = df_label[DLCscorer][bp]['likelihood'].values
        df_bodyparts_x[bpindex, :] = df_label[DLCscorer][bp]['x'].values
        df_bodyparts_y[bpindex, :] = df_label[DLCscorer][bp]['y'].values

    return df_bodyparts_likelihood, df_bodyparts_x, df_bodyparts_y


class PlotBodyparts():

    def __init__(self, path_to_config, case, bodyparts, trainingsetindex=0, shuffle=1):
        """
        Input:
            path_to_config: string
                fullpath to config.yaml file
            case: string
                case number to plot
            bodyparts: list
                A list that contains bodyparts to plot. Each bodypart is in a string format
            shuffle: int, optional
                Integer value specifying the shuffle index to select for training. Default is set to 1
            trainingsetindex: int, optional
                Integer specifying which TrainingsetFraction to use.
                By default the first (note that TrainingFraction is a list in config.yaml).

        """
        self.config = auxiliaryfunctions.read_config(path_to_config)
        self.case = case
        self.bodyparts = bodyparts
        self.shuffle = shuffle
        self.trainingsetindex = trainingsetindex

        (self.df_bodyparts_likelihood, self.df_bodyparts_x, self.df_bodyparts_y) = bodyparts_info(config=self.config,
                                                                                                  case=self.case,
                                                                                                  bodyparts=self.bodyparts,
                                                                                                  trainingsetindex=self.trainingsetindex,
                                                                                                  shuffle=self.shuffle)
        case_full_name = case + '_beh'
        project_path = self.config['project_path']
        path_to_video = os.path.join(
            project_path, 'videos', case_full_name + '.avi')
        self.label_path = os.path.join(
            project_path, 'analysis', case_full_name)
        self.clip = vp(fname=path_to_video)

        # plotting properties
        self._dotsize = 5  # manually change here so that they stand out when plotting
        self._pcutoff = self.config['pcutoff']
        self._colormap = self.config['colormap']
        self._label_colors = get_cmap(len(bodyparts), name=self._colormap)
        self._alphavalue = self.config['alphavalue']
        self._cropping = self.config['cropping']
        if self._cropping:
            self._cropping_coords = [
                self.config['x1'], self.config['x2'], self.config['y1'], self.config['y2']]
        else:
            self._cropping_coords = [
                0, self.clip.width(), 0, self.clip.height()]
        self.nx = self._cropping_coords[1] - self._cropping_coords[0]
        self.ny = self._cropping_coords[3] - self._cropping_coords[2]

    @property
    def dotsize(self):
        return self._dotsize

    @dotsize.setter
    def dotsize(self, value):
        self._dotsize = value

    @property
    def pcutoff(self):
        return self._pcutoff

    @pcutoff.setter
    def pcutoff(self, value):
        self._pcutoff = value

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        if isinstance(value, str):
            self._colormap = value
            self._label_colors = get_cmap(
                len(self.bodyparts), name=self._colormap)
        else:
            raise TypeError("colormap must be in string format")

    @property
    def alphavalue(self):
        return self._alphavalue

    @alphavalue.setter
    def alphavalue(self, value):
        self._alphavalue = value

    # Need to refactorize this portion of the code
    @property
    def cropping(self):
        return self._cropping

    @cropping.setter
    def cropping(self, value):
        if isinstance(value, bool):
            self._cropping = value
            if self._cropping:  # TODO if true, prompt cropping tool box and update cropping_coords
                pass
            else:  # restore cropping coords to the original frame size
                self._cropping_coords = list(
                    0, self.clip.width(), 0, self.clip.height())
        else:
            raise TypeError("cropping must be a boolean")

    @property
    def cropping_coords(self):
        return self._cropping_coords

    # we need to link with cropping tool
    @cropping_coords.setter
    def cropping_coords(self, value):
        if self._cropping:
            if ~isinstance(value, list):
                raise TypeError("coordinates must be a list")
            elif len(value) != 4:
                raise ValueError("length of the coordinates must be 4")
            self._cropping_coords = value
        else:
            print("Cropping is set to False! You cannot reset the cropping coordinates. Default value at original frame size")

    def plot_one_frame(self, frame_num, fit_pupil=False, save_fig=False):

        plt.axis('off')
        fig = plt.figure(frameon=False, figsize=(12, 8))
        # fig = plt.figure(frameon=False, figsize=(nx * 1. / 100, ny * 1. / 100))
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)

        image = self.clip._read_specific_frame(frame_num)

        if self._cropping:
            
            x1 = self._cropping_coords[0]
            x2 = self._cropping_coords[1]            
            y1 = self._cropping_coords[2]
            y2 = self._cropping_coords[3]
            
            image = image[y1:y2, x1:x2]

        plt.imshow(image)

        for bpindex, bp in enumerate(self.bodyparts):
            if self.df_bodyparts_likelihood[bpindex, frame_num] > self.pcutoff:
                plt.scatter(
                    self.df_bodyparts_x[bpindex, frame_num],
                    self.df_bodyparts_y[bpindex, frame_num],
                    s=self.dotsize**2,
                    color=self._label_colors(bpindex),
                    alpha=self.alphavalue
                )

        plt.xlim(0, self.nx)
        plt.ylim(0, self.ny)
        plt.axis('off')
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.gca().invert_yaxis()
        plt.title('frame num: ' + str(frame_num), fontsize=30)

        sm = plt.cm.ScalarMappable(cmap=self._label_colors, norm=plt.Normalize(
            vmin=-0.5, vmax=len(self.bodyparts)-0.5))
        sm._A = []
        cbar = plt.colorbar(sm, ticks=range(len(self.bodyparts)))
        cbar.set_ticklabels(self.bodyparts)
        cbar.ax.tick_params(labelsize=20)

        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(1.0)

        if save_fig:
            plt.tight_layout()
            plt.savefig(os.path.join(
                self.label_path, 'frame_' + str(frame_num) + '.png'))

        plt.close('all')
        return fig

    def plot_over_frames(self, start, end, save_fig=False):
        pass

    def make_gif(self, start, num_frames):
        pass


class FitPupil():

    def __init__(self, *args, **kwargs):
        super(PlotBodyparts, self).__init__(*args, **kwargs)


def plot_over_frames(path_to_config, case, bodyparts2plot, starting, end, shuffle=1, save_fig=False, save_gif=False):
    # TODO change it such that it only plots one frame
    case_full_name = case + '_beh'

    config = auxiliaryfunctions.read_config(path_to_config)

    project_path = config['project_path']
    path_to_video = os.path.join(
        project_path, 'videos', case_full_name + '.avi')
    label_path = os.path.join(project_path, 'analysis', case_full_name)

    trainingsetindex = 0  # modify here as needed
    trainFraction = config['TrainingFraction'][trainingsetindex]
    DLCscorer = auxiliaryfunctions.GetScorerName(
        config, shuffle, trainFraction)

    # obtain video
    clip = vp(fname=path_to_video)

    # label coordinate info
    df_label = pd.read_hdf(os.path.join(
        label_path, case_full_name + DLCscorer + '.h5'))

    df_part, df_part_likelihood, df_part_x, df_part_y = df_part_generator(df_original=df_label,
                                                                          scorer=DLCscorer,
                                                                          parts=bodyparts2plot)
    colors = get_cmap(len(bodyparts2plot), name=colormap)
    # colors_dict = dict(zip(pupil_parts, [colors(i) for i in range(colors.N)]))

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
        fig = plt.figure(frameon=False, figsize=(12, 8))
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
        plt.savefig(os.path.join(
            label_path, 'frame_' + str(starting) + '.png'))

    if save_gif:
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
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
        print("image name is {}".format(
            label_path, 'frame_' + str(starting) + '.png'))
        plt.savefig(os.path.join(
            label_path, 'frame_' + str(starting) + '.png'))

    plt.close('all')
