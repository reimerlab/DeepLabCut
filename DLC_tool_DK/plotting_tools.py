"""
DeepLabCut2.0 Toolbox
D Kim, donniek@bcm.edu
"""

import cv2
import pandas as pd
import os
import numpy as np
import time
import imageio
from collections import OrderedDict
from itertools import cycle
from DLC_tool_DK.cropping_tool import update_inference_cropping_config
from DLC_tool_DK.min_enclosing_circle import smallest_enclosing_circle_naive

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.plotting import get_cmap
# already configured for cv2
from deeplabcut.utils.video_processor import VideoProcessorCV as vp

# for ipython purpose
import pylab as pl
from IPython import display
import matplotlib.pyplot as plt


def get_frame(path_to_video, frame_num):
    cap = cv2.VideoCapture(path_to_video)
    cap.open(path_to_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    _, img = cap.read()
    cap.release()
    return img


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
        self.path_to_config = path_to_config
        self.path_to_cropping_config = path_to_cropping_config

        self.config = auxiliaryfunctions.read_config(self.path_to_config)
        self.cropping_config = auxiliaryfunctions.read_config(
            self.path_to_cropping_config)
        self.case = case
        self.bodyparts = bodyparts  # make it as a property and cascade down all the others
        self.shuffle = shuffle
        self.trainingsetindex = trainingsetindex

        self._case_full_name = case + '_beh'
        self.project_path = self.config['project_path']
        self.path_to_video = os.path.join(
            self.project_path, 'videos', self._case_full_name + '.avi')
        self.path_to_analysis = os.path.join(
            self.project_path, 'analysis, self._case_full_name')

        # inference_case_list = [os.path.basename(os.path.normpath(video_path)).split(
        #     '.')[0] for video_path in list(dict(self.cropping_config).keys())]
        if self.path_to_video not in dict(self.cropping_config).keys():
            raise ValueError(
                "case {} is not added to the cropping_config.yaml yet! \
                    Also make sure you analyze the video without any cropping".format(self.case))

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
        self.df_bodyparts_likelihood = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
            1) == 'likelihood']

        input_cropping = input("cropping for inference ? True/False: ")

        if input_cropping.casefold() == 'true' or input_cropping.casefold() == 't' \
                or input_cropping.casefold() == 'yes' or input_cropping.casefold() == 'y':
            self._cropping = True

            input_use_preset_cropping_coords = input(
                "Use predefined cropping coordinates from cropping_config file (True) "
                "or define new cropping coordinates (False)? : ")

            if input_use_preset_cropping_coords.casefold() != 'true' and input_use_preset_cropping_coords.casefold() != 't' \
                    and input_use_preset_cropping_coords.casefold() != 'yes' and input_use_preset_cropping_coords.casefold() != 'y':

                self._use_preset_cropping_coords = False

                print(
                    "This function cannot be called in Jupyter Notebook! Do it in IPython!")
                print("Now the cropping coordinates are updated in cropping_config.yaml")

                update_inference_cropping_config(
                    cropping_config=self.path_to_cropping_config, video_path=self.path_to_video)

                # re-read cropping config to get the updated coords
                self.cropping_config = auxiliaryfunctions.read_config(
                    self.path_to_cropping_config)

            else:
                self._use_preset_cropping_coords = True

            self._cropping_coords = list(
                dict(self.cropping_config[self.path_to_video]).values())[1:]
            self.df_bodyparts_x = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                1) == 'x'] - self.cropping_coords[0]
            self.df_bodyparts_y = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                1) == 'y'] - self.cropping_coords[2]

        else:
            self._cropping = False
            self._cropping_coords = [
                0, self.clip.width(), 0, self.clip.height()]
            self.df_bodyparts_x = self._orig_df_bodyparts.iloc[:,
                                                               self._orig_df_bodyparts.columns.get_level_values(1) == 'x']
            self.df_bodyparts_y = self._orig_df_bodyparts.iloc[:,
                                                               self._orig_df_bodyparts.columns.get_level_values(1) == 'y']

        self.nx = self._cropping_coords[1] - self._cropping_coords[0]
        self.ny = self._cropping_coords[3] - self._cropping_coords[2]

        # plotting properties
        self._dotsize = 5
        self._line_thickness = 1
        self._pcutoff = self.config['pcutoff']
        self._colormap = self.config['colormap']
        self._label_colors = get_cmap(len(bodyparts), name=self._colormap)
        self._alphavalue = self.config['alphavalue']
        self._fig_size = [12, 8]
        self._dpi = 100

        self.tf_likelihood_array = self.df_bodyparts_likelihood.values > self._pcutoff

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

    @property
    def fig_size(self):
        return self._fig_size

    @fig_size.setter
    def fig_size(self, value):
        if isinstance(value, list):
            self._fig_size = value
        else:
            raise TypeError("fig_size must be in a list format")

    @property
    def dpi(self):
        return self._dpi

    @dpi.setter
    def dpi(self, value):
        self._dpi = value

    @property
    def cropping(self):
        return self._cropping

    @cropping.setter
    def cropping(self, value):
        if isinstance(value, bool):
            self._cropping = value
            if self._cropping:

                input_use_preset_cropping_coords = input(
                    "Use predefined cropping coordinates from cropping_config file (True) "
                    "or define new cropping coordinates (False)? : ")

                if input_use_preset_cropping_coords.casefold() != 'true' and input_use_preset_cropping_coords.casefold() != 't' \
                        and input_use_preset_cropping_coords.casefold() != 'yes' and input_use_preset_cropping_coords.casefold() != 'y':

                    self._use_preset_cropping_coords = True

                    print(
                        "This function cannot be called in Jupyter Notebook! Do it in IPython!")

                    update_inference_cropping_config(
                        cropping_config=self.path_to_cropping_config, video_path=self.path_to_video)

                    print(
                        "Now the cropping coordinates are updated in cropping_config.yaml")
                    # re-read cropping config to get the updated coords
                    self.cropping_config = auxiliaryfunctions.read_config(
                        self.path_to_cropping_config)

                else:
                    self._use_preset_cropping_coords = False

                self._cropping_coords = list(
                    dict(self.cropping_config[self.path_to_video]).values())[1:]
                self.df_bodyparts_x = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                    1) == 'x'] - self.cropping_coords[0]
                self.df_bodyparts_y = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                    1) == 'y'] - self.cropping_coords[2]

            else:  # restore cropping coords to the original frame size
                self._cropping_coords = [
                    0, self.clip.width(), 0, self.clip.height()]
                self.df_bodyparts_x = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                    1) == 'x']
                self.df_bodyparts_y = self._orig_df_bodyparts.iloc[:, self._orig_df_bodyparts.columns.get_level_values(
                    1) == 'y']
        else:
            raise TypeError("cropping must be a boolean")

        self.nx = self._cropping_coords[1] - self._cropping_coords[0]
        self.ny = self._cropping_coords[3] - self._cropping_coords[2]

    @property
    def cropping_coords(self):
        return self._cropping_coords

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

    def configure_plot(self):
        fig = plt.figure(frameon=False, figsize=self.fig_size)
        ax = fig.add_subplot(1, 1, 1)
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)
        plt.xlim(0, self.nx)
        plt.ylim(0, self.ny)

        plt.gca().invert_yaxis()

        sm = plt.cm.ScalarMappable(cmap=self._label_colors, norm=plt.Normalize(
            vmin=-0.5, vmax=len(self.bodyparts)-0.5))
        sm._A = []
        cbar = plt.colorbar(sm, ticks=range(len(self.bodyparts)))
        cbar.set_ticklabels(self.bodyparts)
        cbar.ax.tick_params(labelsize=18)

        return fig, ax

    def plot_core(self, fig, ax, frame_num):
        # it's given in 3 channels but every channel is the same i.e. grayscale
        image = self.clip._read_specific_frame(frame_num)

        if self.cropping:

            x1 = self.cropping_coords[0]
            x2 = self.cropping_coords[1]
            y1 = self.cropping_coords[2]
            y2 = self.cropping_coords[3]

            image = image[y1:y2, x1:x2]

        ax_frame = ax.imshow(image, cmap='gray')

        # plot bodyparts above the pcutoff
        bpindex, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)
        ax_scatter = ax.scatter(df_x_coords.values, df_y_coords.values, s=self.dotsize**2,
                                color=self._label_colors(bpindex), alpha=self.alphavalue)

        return {'ax_frame': ax_frame, 'ax_scatter': ax_scatter}

    def plot_one_frame(self, frame_num, save_fig=False):

        fig, ax = self.configure_plot()

        ax_dict = self.plot_core(fig, ax, frame_num)

        plt.axis('off')
        plt.tight_layout()
        plt.title('frame num: ' + str(frame_num), fontsize=30)

        fig.canvas.draw()

        if save_fig:
            plt.savefig(os.path.join(
                self.label_path, 'frame_' + str(frame_num) + '.png'))

        # return ax_dict

    def plot_multi_frames(self, start, end, save_gif=False):

        plt_list = []

        fig, ax = self.configure_plot()

        for frame_num in range(start, end):
            ax_dict = self.plot_core(fig, ax, frame_num)

            plt.axis('off')
            plt.tight_layout()
            plt.title('frame num: ' + str(frame_num), fontsize=30)

            fig.canvas.draw()

            data = np.fromstring(fig.canvas.tostring_rgb(),
                                 dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt_list.append(data)

            display.clear_output(wait=True)
            display.display(pl.gcf())
            time.sleep(0.5)

            plt.cla()

        if save_gif:
            gif_name = self.case + \
                str(start) + '_' + str(end) + '.gif'
            save_dir = os.path.join(self.label, gif_name)
            imageio.mimsave(save_dir, plt_list, fps=1)

        plt.close('all')


class PupilFitting(PlotBodyparts):
    # for this class, all bodyparts must be provided... so why bother providing bodyparts as input?
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
        super().__init__(path_to_config, path_to_cropping_config,
                         case, bodyparts, trainingsetindex=0, shuffle=1)

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
            A dictionary containing the fitted frame and its corresponding binary mask.
            For each key in dictionary:
                frame:
                    A numpy array frame with eyelids connected
                mask:
                    A binary numpy array
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

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
                np.array(frame), coord_0, coord_1, color=(255, 0, 0), thickness=self.line_thickness)
            mask = cv2.line(
                mask, coord_0, coord_1, color=(255), thickness=self.line_thickness)

        # fill out the mask with 1s OUTSIDE of the mask, then invert 0 and 1
        # for cv2.floodFill, need a mask that is 2 pixels bigger than the input image
        new_mask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
        cv2.floodFill(mask, new_mask, seedPoint=(0, 0), newVal=124)

        final_mask = np.logical_not(new_mask).astype(int)[1:-1, 1:-1]

        # ax.imshow(mask)
        return {'frame': frame, 'mask': final_mask}

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
            not occur, return the original frame with center and raidus as None. 
            For each key in dictionary:
                frame: a numpy array of the frame with pupil circle
                center: coordinates of the center of the fitted circle. In tuple format
                radius: radius of the fitted circle in int format
                pupil_label_num: number of pupil labels used for fitting
                mask: a binary mask for the fitted circle area
        """

        mask = np.zeros(frame.shape, dtype=np.uint8)

        _, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)

        pupil_labels = [label for label in list(
            df_x_coords.index.get_level_values(0)) if 'pupil' in label]

        if len(pupil_labels) <= 2:
            print('Frame number: {} has only 2 or less pupil label. Skip fitting!'.format(
                frame_num))
            center = None
            radius = None

        elif len(pupil_labels) > 2:
            pupil_x = df_x_coords.loc[pupil_labels].values
            pupil_y = df_y_coords.loc[pupil_labels].values

            pupil_coords = list(zip(pupil_x, pupil_y))

            x, y, radius = smallest_enclosing_circle_naive(pupil_coords)

            center = (int(x), int(y))
            radius = int(radius)

            # opencv has some issues with dealing with np objects. Cast it manually again
            frame = cv2.circle(np.array(frame), center,
                               radius, color=(0, 255, 0), thickness=self.line_thickness)

            mask = cv2.circle(mask, center,
                              radius, color=(0, 255, 0), thickness=self.line_thickness)

        # fill out the mask with 1s OUTSIDE of the mask, then invert 0 and 1
        # for cv2.floodFill, need a mask that is 2 pixels bigger than the input image
        new_mask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
        cv2.floodFill(mask, new_mask, seedPoint=(0, 0), newVal=1)
        final_mask = np.logical_not(new_mask).astype(int)[1:-1, 1:-1]

        return {'frame': frame, 'center': center, 'radius': radius, 'pupil_label_num': len(pupil_labels), 'mask': final_mask}

    def detect_visible_pupil(self, frame_num, frame):
        """
        Given a frame, find a visible part of the pupil by finding the intersection of pupil and eyelid masks
        If pupil mask does not exist(i.e. label < 3), return None

        Input:
            frame_num: int
                frame number to extract a specific frame
            frame:

        Output:
            binary numpy array or None
                If pupil was fitted, then return the visible part. Otherwise None
        """

        eyelid_connected = self.connect_eyelids(frame_num, frame)
        pupil_fitted = self.fit_circle_to_pupil(
            frame_num, eyelid_connected['frame'])

        if pupil_fitted['pupil_label_num'] >= 3:
            return np.logical_and(pupil_fitted['mask'], eyelid_connected['mask']).astype(int)
        else:
            return None

    def fitted_plot_core(self, fig, ax, frame_num):
        # it's given in 3 channels but every channel is the same i.e. grayscale

        image = self.clip._read_specific_frame(frame_num)

        if self._cropping:

            x1 = self._cropping_coords[0]
            x2 = self._cropping_coords[1]
            y1 = self._cropping_coords[2]
            y2 = self._cropping_coords[3]

            image = image[y1:y2, x1:x2]

        # plot bodyparts above the pcutoff
        bpindex, x_coords, y_coords = self.coords_pcutoff(frame_num)
        ax_scatter = ax.scatter(x_coords.values, y_coords.values, s=self.dotsize**2,
                                color=self._label_colors(bpindex), alpha=self.alphavalue)

        eyelid_connected = self.connect_eyelids(frame_num, frame=image)

        pupil_fitted = self.fit_circle_to_pupil(
            frame_num, frame=eyelid_connected['frame'])

        ax_frame = ax.imshow(pupil_fitted['frame'])

        color_mask = np.zeros(shape=image.shape, dtype=np.uint8)
        if pupil_fitted['pupil_label_num'] >= 3:
            visible_mask = np.logical_and(
                pupil_fitted['mask'], eyelid_connected['mask']).astype(int)

            # 126,0,255 for the color
            color_mask[visible_mask == 1, 0] = 126
            color_mask[visible_mask == 1, 2] = 255

        ax_mask = ax.imshow(color_mask, alpha=0.3)

        return {'ax_frame': ax_frame, 'ax_scatter': ax_scatter, 'ax_mask': ax_mask}

    def plot_fitted_frame(self, frame_num, save_fig=False):

        fig, ax = self.configure_plot()
        ax_dict = self.fitted_plot_core(fig, ax, frame_num)

        plt.title('frame num: ' + str(frame_num), fontsize=30)

        # plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()

        if save_fig:
            plt.savefig(os.path.join(
                self.label_path, 'fitted_frame_' + str(frame_num) + '.png'))

    def plot_fitted_multi_frames(self, start, end, save_gif=False):

        fig, ax = self.configure_plot()

        plt_list = []

        for frame_num in range(start, end):

            ax_dict = self.fitted_plot_core(fig, ax, frame_num)

            plt.axis('off')
            plt.tight_layout()
            plt.title('frame num: ' + str(frame_num), fontsize=30)

            fig.canvas.draw()

            data = np.fromstring(fig.canvas.tostring_rgb(),
                                 dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt_list.append(data)

            display.clear_output(wait=True)
            display.display(pl.gcf())
            time.sleep(0.5)

            plt.cla()

        if save_gif:
            gif_name = self.case + '_fitted_' + \
                str(start) + '_' + str(end) + '.gif'
            save_dir = os.path.join(self.label_path, gif_name)
            imageio.mimsave(save_dir, plt_list, fps=1)

        plt.close('all')

    def make_movie(self, start, end):

        import matplotlib.animation as animation

        # initlize with first frame
        fig, ax = self.configure_plot()
        ax_dict = self.fitted_plot_core(fig, ax, frame_num=start)

        plt.axis('off')
        plt.tight_layout()
        plt.title('frame num: ' + str(frame_num), fontsize=30)

        def update_frame(frame_num):
            ax_dict['frame']

        #
        ani = animation.FuncAnimation(fig, update_frame, range(
            start+1, end), interval=int(1/self.clip.FPS))
        # ani = animation.FuncAnimation(fig, self.plot_fitted_frame, 10)
        writer = animation.writers['ffmpeg']

        ani.save('demo.avi', writer=writer, dpi=self.dpi)
        return ani


# TODO build a classifier for 3 cases of eyes: closed, blurry, and open


class EyeStatus():
    def __init__():
        pass
