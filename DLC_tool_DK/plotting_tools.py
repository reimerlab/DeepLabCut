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
from collections import OrderedDict
from itertools import cycle
from DLC_tool_DK.cropping_tool import update_inference_cropping_config

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
    Given bodyparts, return corresponding likelihood, x-coordinates, and y-coordinates in dataframe

    Using pandas instead of numpy as my data is in range of 50k to 500k
    http://gouthamanbalaraman.com/blog/numpy-vs-pandas-comparison.html
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

    df_bodyparts_likelihood = df_bodyparts.iloc[:, df_bodyparts.columns.get_level_values(
        1) == 'likelihood']
    df_bodyparts_x = df_bodyparts.iloc[:,
                                       df_bodyparts.columns.get_level_values(1) == 'x']
    df_bodyparts_y = df_bodyparts.iloc[:,
                                       df_bodyparts.columns.get_level_values(1) == 'y']

    return df_bodyparts_likelihood, df_bodyparts_x, df_bodyparts_y


class PlotBodyparts():

    def __init__(self, path_to_config, path_to_cropping_config, case, bodyparts, trainingsetindex=0, shuffle=1):
        """
        Input:
            path_to_config: string
                fullpath to config.yaml file
            path_to_cropping_config: string
                fullpath to cropping_config.yaml file
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
        self.cropping_config = auxiliaryfunctions.read_config(
            path_to_cropping_config)
        self.case = case
        self.bodyparts = bodyparts  # make it as a property and cascade down all the others
        self.shuffle = shuffle
        self.trainingsetindex = trainingsetindex

        self._case_full_name = case + '_beh'
        self.project_path = self.config['project_path']
        self.path_to_video = os.path.join(
            self.project_path, 'videos', self._case_full_name + '.avi')

        #TODO compare by case, not the fullpath to the video
        if self.path_to_video not in self.cropping_config.keys():
            raise ValueError(
                "case {} is not added to the cropping_config.yaml yet! \
                    Also make sure you analyze the video without any cropping".format(self.path_to_video))

        self.label_path = os.path.join(
            self.project_path, 'analysis', self._case_full_name)
        self.clip = vp(fname=self.path_to_video)

        self._trainFraction = self.config['TrainingFraction'][self.trainingsetindex]
        self._DLCscorer = auxiliaryfunctions.GetScorerName(
            self.config, self.shuffle, self._trainFraction)

        self.df_label = pd.read_hdf(os.path.join(
            self.label_path, self._case_full_name + self._DLCscorer + '.h5'))

        # TODO maybe rename the variable
        self._orig_df_bodyparts = self.df_label[self._DLCscorer][self.bodyparts]
        self.df_bodyparts_likelihood = df_bodyparts.iloc[:, df_bodyparts.columns.get_level_values(
            1) == 'likelihood']

        self._cropping = input("cropping for inference ? True/False: ")
        if self._cropping.casefold() == 'True' or self._cropping.casefold() == 'T' or self._cropping.casefold() == 'yes' or self._cropping.casefold() == 'y':
            update_inference_cropping_config(
                cropping_config=self.cropping_config, video_path=self.path_to_video)
            self._cropping_coords = list(
                dict(self.cropping_config[self.path_to_video]).values())[1:]
            self._df_bodyparts_x = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                1) == 'x'] - self.cropping_coords[0]
            self._df_bodyparts_y = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                1) == 'y'] - self.cropping_coords[2]

        else:
            self._cropping_coords = [
                0, self.clip.width(), 0, self.clip.height()]
            self._df_bodyparts_x = self._orig_df_bodyparts.iloc[:,
                                                                self._orig_df_bodyparts.columns.get_level_values(1) == 'x']
            self._df_bodyparts_y = self._orig_df_bodyparts.iloc[:,
                                                                self._orig_df_bodyparts.columns.get_level_values(1) == 'y']

        self.nx = self._cropping_coords[1] - self._cropping_coords[0]
        self.ny = self._cropping_coords[3] - self._cropping_coords[2]

        self.tf_likelihood_array = self.df_bodyparts_likelihood.values > self.pcutoff

        # plotting properties
        self._dotsize = 5
        self._line_thickness = 1
        self._pcutoff = self.config['pcutoff']
        self._colormap = self.config['colormap']
        self._label_colors = get_cmap(len(bodyparts), name=self._colormap)
        self._alphavalue = self.config['alphavalue']

    @property
    def dotsize(self):
        return self._dotsize

    @dotsize.setter
    def dotsize(self, value):
        self._dotsize = value

    @property
    def line_thickness(self):
        return self._line_thickness

    @line_thickness.setter
    def line_thickness(self, value):
        if isinstance(value, int):
            self._line_thickness = value

        else:
            raise TypeError("line thickness must be integer")

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
            if self._cropping:
                print("This function cannot be called in Jupyter Notebook")
                update_inference_cropping_config(
                    cropping_config=self.cropping_config, video_path=self.path_to_video)
                self._cropping_coords = list(
                    dict(self.cropping_config[self.path_to_video]).values())[1:]
                self._df_bodyparts_x = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                    1) == 'x'] - self.cropping_coords[0]
                self._df_bodyparts_y = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                    1) == 'y'] - self.cropping_coords[2]

            else:  # restore cropping coords to the original frame size
                self._cropping_coords = list(
                    0, self.clip.width(), 0, self.clip.height())
                self._df_bodyparts_x = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                    1) == 'x']
                self._df_bodyparts_y = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                    1) == 'y']
        else:
            raise TypeError("cropping must be a boolean")

    @property
    def cropping_coords(self):
        return self._cropping_coords

    # @property
    # def bodyparts_info(self):
    #     return self._df_bodyparts_likelihood, self._df_bodyparts_x, self._df_bodyparts_y

    # @bodyparts_info.setter
    # def bodyparts_info(self):
    #     """
    #     Given bodyparts, return corresponding likelihood, x-coordinates, and y-coordinates in dataframe

    #     Using pandas instead of numpy as my data is in range of 50k to 500k
    #     http://gouthamanbalaraman.com/blog/numpy-vs-pandas-comparison.html
    #     """
    #     self._trainFraction = self.config['TrainingFraction'][self.trainingsetindex]
    #     self._DLCscorer = auxiliaryfunctions.GetScorerName(
    #         self.config, self.shuffle, trainFraction)

    #     self.df_label = pd.read_hdf(os.path.join(
    #         self.label_path, self._case_full_name + self._DLCscorer + '.h5'))

    #     df_bodyparts = self.df_label[self._DLCscorer][self.bodyparts]

    #     df_bodyparts_likelihood = df_bodyparts.iloc[:, df_bodyparts.columns.get_level_values(
    #         1) == 'likelihood']
    #     df_bodyparts_x = df_bodyparts.iloc[:,
    #                                        df_bodyparts.columns.get_level_values(1) == 'x']
    #     df_bodyparts_y = df_bodyparts.iloc[:,
    #                                        df_bodyparts.columns.get_level_values(1) == 'y']
    #     if self.cropping:
    #         df_bodyparts_x -= self.cropping_coords[0]
    #         df_bodyparts_y -= self.cropping_coords[2]

    #     return df_bodyparts_likelihood, df_bodyparts_x, df_bodyparts_y

    def coords_pcutoff(self, frame_num):
        """
        Given a frame number, return bpindex, x & y coordinates that meet pcutoff criteria
        Input:
            frame_num: int
                A desired frame number
        Output:
            bpindex: list
                A list of integers that match with bodypart. For instance, if the bodypart is ['A','B','C'] 
                and only 'A' and 'C'qualifies the pcutoff, then bpindex = [0,2]
            x_coords: pandas series
                A pandas series that contains coordinates whose values meet pcutoff criteria
            y_coords: pandas series
                A pandas series that contains coordinates whose values meet pcutoff criteria
        """
        frame_num_tf = self.tf_likelihood_array[frame_num, :]
        bpindex = [i for i, x in enumerate(frame_num_tf) if x]

        df_x_coords = self.df_bodyparts_x.loc[frame_num, :][bpindex]
        df_y_coords = self.df_bodyparts_y.loc[frame_num, :][bpindex]

        return bpindex, df_x_coords, df_y_coords

    def plot_one_frame(self, frame_num, save_fig=False, save_gif=False):

        plt.axis('off')
        fig = plt.figure(frameon=False, figsize=(10, 8))
        # fig = plt.figure(frameon=False, figsize=(nx * 1. / 100, ny * 1. / 100))
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)

        image = self.clip._read_specific_frame(frame_num)

        if self.cropping:

            x1 = self.cropping_coords[0]
            x2 = self.cropping_coords[1]
            y1 = self.cropping_coords[2]
            y2 = self.cropping_coords[3]

            image = image[y1:y2, x1:x2]

        plt.imshow(image)

        # plot bodyparts above the pcutoff
        bpindex, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)
        plt.scatter(df_x_coords.values, df_y_coords.values, s=self.dotsize**2,
                    color=self._label_colors(bpindex), alpha=self.alphavalue)

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

        if save_gif:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            return image

        # plt.show()
        # plt.close('all')

        return fig

    def plot_over_frames(self, start, end, save_fig=False):

        for i in range(start, end):
            self.plot_one_frame(i, save_fig=save_fig)

        # plt.close('all')

    def make_gif(self, start, num_frames, save_gif=True):
        gif_name = self.case + '_' + \
            str(start) + '_' + str(start+num_frames) + '.gif'
        save_dir = os.path.join(
            self.config['project_path'], 'analysis', self.case + '_beh', gif_name)

        plt.figure(frameon=False, figsize=(10, 8))
        plt_list = []

        for i in range(start, start + num_frames):
            plot = self.plot_one_frame(frame_num=i, save_gif=True)
            plt_list.append(plot)
        imageio.mimsave(save_dir, plt_list, fps=1)


class PupilFitting(PlotBodyparts):
    # for this class, all bodyparts must be provided... so why bother providing bodyparts as input?
    def __init__(self, path_to_config, case, bodyparts, trainingsetindex=0, shuffle=1):
        super().__init__(path_to_config, case, bodyparts, trainingsetindex=0, shuffle=1)

        self.complete_pupil_labels = [
            part for part in bodyparts if part.startswith('pupil')]
        self.complete_eyelid_labels = [
            part for part in bodyparts if part.startswith('eyelid')]
        self.complete_eyelid_graph = {'eyelid_top': 'eyelid_top_right',
                                      'eyelid_top_right': 'eyelid_right',
                                      'eyelid_right': 'eyelid_right_bottom',
                                      'eyelid_right_bottom': 'eyelid_bottom',
                                      'eyelid_bottom': 'eyelid_bottom_left',
                                      'eyelid_bottom_left': 'eyelid_left',
                                      'eyelid_left': 'eyelid_left_top',
                                      'eyelid_left_top': 'eyelid_top'}

    def old_plot_fitted_frame(self, frame_num, save_fig=False, save_gif=False):
        pass

    def coords_pcutoff(self, frame_num):
        """
        Given a frame number, return bpindex, x & y coordinates that meet pcutoff criteria
        Input:
            frame_num: int
                A desired frame number
        Output:
            bpindex: list
                A list of integers that match with bodypart. For instance, if the bodypart is ['A','B','C'] 
                and only 'A' and 'C'qualifies the pcutoff, then bpindex = [0,2]
            x_coords: pandas series
                A pandas series that contains coordinates whose values meet pcutoff criteria
            y_coords: pandas series
                A pandas series that contains coordinates whose values meet pcutoff criteria
        """
        frame_num_tf = self.tf_likelihood_array[frame_num, :]
        bpindex = [i for i, x in enumerate(frame_num_tf) if x]

        df_x_coords = self.df_bodyparts_x.loc[frame_num, :][bpindex]
        df_y_coords = self.df_bodyparts_y.loc[frame_num, :][bpindex]

        return bpindex, df_x_coords, df_y_coords

    def connect_eyelids(self, frame_num, frame):
        """
        connect eyelid labels with a straight line. If a label is missing, do not connect and skip to the next label.
        Input:
            frame_num: int
                A desired frame number
            frame: numpy array
                A frame to be fitted
        Output:
            frame:
                A numpy array frame with eyelids connected
        """

        _, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)
        eyelid_labels = [label for label in list(
            df_x_coords.index.get_level_values(0)) if 'eyelid' in label]

        for eyelid in eyelid_labels:
            next_bp = self.complete_eyelid_graph[eyelid]

            if next_bp not in eyelid_labels:
                continue

            coord_0 = tuple(
                map(int, map(round, [df_x_coords[eyelid].values[0], df_y_coords[eyelid].values[0]])))
            coord_1 = tuple(
                map(int, map(round, [df_x_coords[next_bp].values[0], df_y_coords[next_bp].values[0]])))
            # opencv has some issues with dealing with np objects. Cast it manually again
            frame = cv2.line(
                np.array(frame), coord_0, coord_1, color=(0, 255, 0), thickness=self.line_thickness)

        return frame

    def fit_circle_to_pupil(self, frame_num, frame):
        """
        Fit a circle to the pupil
        Input:
            frame_num: int
                A desired frame number
            frame: numpy array
                A frame to be fitted
        Output: dictionary
            A dictionary with the fitted frame, center and radius of the fitted circle. If fitting did
            not occur, return the original frame with center and raidus as None
        """

        _, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)

        pupil_labels = [label for label in list(
            df_x_coords.index.get_level_values(0)) if 'pupil' in label]

        print(pupil_labels)

        if len(pupil_labels) <= 1:
            print('Frame number: {} has only 1 or less pupil label. Skip fitting!'.format(
                frame_num))
            center = None
            radius = None

        elif len(pupil_labels) == 2:
            print('havent implemented yet!')
            center = None
            radius = None

        elif len(pupil_labels) >= 3:
            pupil_x = df_x_coords.loc[pupil_labels].values
            pupil_y = df_y_coords.loc[pupil_labels].values

            pupil_coords = np.array(zip)

            pupil_coords = np.array(
                list(zip(pupil_x, pupil_y))).reshape(-1, 1, 2).astype(int)

            (x, y), radius = cv2.minEnclosingCircle(pupil_coords)
            center = (int(x), int(y))
            radius = int(radius)
            # opencv has some issues with dealing with np objects. Cast it manually again
            frame = cv2.circle(np.array(frame), center,
                               radius, color=(0, 255, 0), thickness=self.line_thickness)

        return {'frame': frame, 'center': center, 'radius': radius}

    def plot_fitted_frame(self, frame_num, save_fig=False, save_gif=False):

        plt.axis('off')
        fig = plt.figure(frameon=False, figsize=(10, 8))
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

        # plot bodyparts above the pcutoff
        bpindex, x_coords, y_coords = self.coords_pcutoff(frame_num)
        plt.scatter(x_coords.values, y_coords.values, s=self.dotsize**2,
                    color=self._label_colors(bpindex), alpha=self.alphavalue)

        eyelid_connected_frame = self.connect_eyelids(frame_num, frame=image)

        circle_fit = self.fit_circle_to_pupil(
            frame_num, frame=eyelid_connected_frame)

        plt.imshow(circle_fit['frame'])
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

        if save_gif:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close('all')
            return image

        # plt.close('all')
        return fig
