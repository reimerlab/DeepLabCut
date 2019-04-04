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
import datajoint as dj

from DLC_tool_DK.cropping_tool import update_inference_cropping_config
from DLC_tool_DK.plotting_tools import PlotBodyparts, PupilFitting
from DLC_tool_DK.segmentation_score import compute_segmentation_score

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.plotting import get_cmap
# already configured for cv2
from deeplabcut.utils.video_processor import VideoProcessorCV as vp

# for ipython purpose
import pylab as pl
from IPython import display


def key_dict_generater(case):
    case_key = {'animal_id': None, 'session': None, 'scan_idx': None}
    for ind, key in enumerate(case_key.keys()):
        case_key[key] = int(case.split('_')[ind])

    return case_key


pupil_table = dj.create_virtual_module('pupil_table', 'pipeline_eye')


class CompareFittingMethod(PupilFitting):

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

        if 'compressed_cropped' in self.case:
            self._cropping_coords = list(
                dict(self.cropping_config[self.path_to_video]).values())[1:]

        case_key = key_dict_generater(self.case)
        old_center, old_diameter = (
            pupil_table.FittedContour.Ellipse & case_key).fetch('center', 'major_r')

        old_contour = (pupil_table.ManuallyTrackedContours.Frame &
                       case_key).fetch('contour')

        # update the center and radius of old method wrt to cropped coordinates
        self.old_center = []
        for coords in old_center:
            if coords is not None:
                self.old_center.append(tuple((np.round(coords).astype(
                    np.int32) - np.array([self.cropping_coords[0], self.cropping_coords[2]])).tolist()))
            else:
                self.old_center.append(coords)

        self.old_diameter = [np.round(rad).astype(
            np.int32) if rad is not None else rad for rad in old_diameter]

        self.old_contour = []
        for contour in old_contour:
            if contour is not None:
                self.old_contour.append(contour.squeeze(
                ) - np.array([self.cropping_coords[0], self.cropping_coords[2]]))
            else:
                self.old_contour.append(contour)

    def old_fit_circle_to_pupil(self, frame_num, frame):
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
        # TODO Might need to refactor mask and final_mask

        mask = np.zeros(frame.shape, dtype=np.uint8)
        final_mask = mask
        if self.old_center[frame_num] is not None:
            frame = cv2.circle(np.array(frame), center=self.old_center[frame_num], radius=int(self.old_diameter[frame_num]/2),
                               color=(0, 0, 255), thickness=1)
            mask = cv2.circle(np.array(mask), center=self.old_center[frame_num], radius=int(self.old_diameter[frame_num]/2),
                              color=(0, 0, 255), thickness=1)

            new_mask = np.zeros(
                (mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
            cv2.floodFill(mask, new_mask, seedPoint=(0, 0), newVal=1)
            final_mask = np.logical_not(new_mask).astype(int)[1:-1, 1:-1]

        return {'frame': frame, 'mask': final_mask}

    def compare_fitted_plot_core(self, fig, ax, frame_num, only_circles=False):
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

        if only_circles:
            pupil_fitted = self.fit_circle_to_pupil(
                frame_num, frame=image)
            old_pupil_fitted = self.old_fit_circle_to_pupil(
                frame_num=frame_num, frame=pupil_fitted['frame'])
                
            ax_frame = ax.imshow(old_pupil_fitted['frame'])

            return {'ax_frame': ax_frame}
            

        if not only_circles:
            ax_scatter = ax.scatter(x_coords.values, y_coords.values, s=self.dotsize**2,
                                    color=self._label_colors(bpindex), alpha=self.alphavalue)

            eyelid_connected = self.connect_eyelids(frame_num, frame=image)

            pupil_fitted = self.fit_circle_to_pupil(
                frame_num, frame=eyelid_connected['frame'])

            old_pupil_fitted = self.old_fit_circle_to_pupil(
                frame_num=frame_num, frame=pupil_fitted['frame'])

            ax_frame = ax.imshow(old_pupil_fitted['frame'])

            color_mask = np.zeros(shape=image.shape, dtype=np.uint8)
            if pupil_fitted['pupil_label_num'] >= 3:
                visible_mask = np.logical_and(
                    pupil_fitted['mask'], eyelid_connected['mask']).astype(int)

                # 126,0,255 for the color
                color_mask[visible_mask == 1, 0] = 126
                color_mask[visible_mask == 1, 2] = 255

                # plot center
                ax.scatter(pupil_fitted['center'][0], pupil_fitted['center']
                        [1], color='lime', label='DLC circle')

            ax_mask = ax.imshow(color_mask, alpha=0.3)

            if self.old_center[frame_num] is not None:

                ax_contour_scatter = ax.scatter(
                    self.old_contour[frame_num][:,
                                                0], self.old_contour[frame_num][:, 1],
                    color=[51./255, 51./255, 0], alpha=1, s=10, label='old contour')

                # in matplotlib, colors must be given between 0 and 1 in RGB order

                ax.scatter(self.old_center[frame_num][0], self.old_center[frame_num]
                        [1], color='blue', label='non-DLC circle')

                ax.legend(loc='upper left')
            else:
                ax_contour_scatter = ax.scatter([], [])

            return {'ax_frame': ax_frame, 'ax_scatter': ax_scatter, 'ax_contour_scatter': ax_contour_scatter, 'ax_mask': ax_mask}

    def plot_compare_fitted_frame(self, frame_num, only_circles=False, save_fig=False):

        fig, ax = self.configure_plot()
        _ = self.compare_fitted_plot_core(fig, ax, frame_num, only_circles)

        plt.title('frame num: ' + str(frame_num), fontsize=30)

        plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()

        if save_fig:
            plt.savefig(os.path.join(
                self.label_path, 'fitted_frame_' + str(frame_num) + '.png'))

    def plot_compare_fitted_multi_frames(self, start, end, only_circles=False, save_gif=False):

        fig, ax = self.configure_plot()

        plt_list = []

        for frame_num in range(start, end):

            _ = self.compare_fitted_plot_core(fig, ax, frame_num, only_circles)

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

    def make_movie(self, start, end, save_as_avi=False, save_as_gif=False):

        if save_as_gif:
            assert (end-start) < 30, "If more than 30 frames, make it into avi, not gif!"

        import matplotlib.animation as animation

        # initlize with start frame
        fig, ax = self.configure_plot()
        # ax_dict = self.fitted_plot_core(fig, ax, frame_num=start)
        _ = self.compare_fitted_plot_core(fig, ax, frame_num=start)

        plt.axis('off')
        plt.tight_layout()
        plt.title('frame num: ' + str(start), fontsize=30)

        def update_frame(frame_num):

            # clear out the axis
            plt.cla()
            # new_ax_dict = self.fitted_plot_core(fig, ax, frame_num=frame_num)
            _ = self.compare_fitted_plot_core(fig, ax, frame_num=frame_num)

            plt.axis('off')
            plt.tight_layout()
            plt.title('frame num: ' + str(frame_num), fontsize=30)

        ani = animation.FuncAnimation(fig, update_frame, range(
            start+1, end))  # , interval=int(1/self.clip.FPS)
        # ani = animation.FuncAnimation(fig, self.plot_fitted_frame, 10)

        if save_as_avi:
            writer = animation.writers['ffmpeg'](fps=self.clip.FPS)

            # dpi=self.dpi, fps=self.clip.FPS
            video_name = os.path.join(
                self.path_to_analysis, self._case_full_name + '_labeled.avi')
            ani.save(video_name, writer=writer, dpi=self.dpi)

        if save_as_gif:
            writer = animation.writers['imagemagick'](fps=1)

            gif_name = os.path.join(
                self.path_to_analysis, self._case_full_name + '_{}_{}.gif'.format(start, end))
            ani.save(gif_name, writer=writer, dpi=self.dpi)

        return ani

    def segmentation_score(self):
        """
        Compute segmentation scores of two methods (DLC vs non-DLC)

        Output: pandas dataframe
            pandas dataframe containing scores
        """

        score_mat = np.zeros(shape=(self.clip.nframes, 4))
        # score_mat = np.zeros(shape=(1000, 4))

        for frame_num in range(self.clip.nframes):
        # for frame_num in range(score_mat.shape[0]):

            _, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)

            pupil_labels = [label for label in list(
                df_x_coords.index.get_level_values(0)) if 'pupil' in label]

            if self.old_center[frame_num] is not None and len(pupil_labels) > 2:

                image = self.clip._read_specific_frame(frame_num)

                if self._cropping:

                    x1 = self._cropping_coords[0]
                    x2 = self._cropping_coords[1]
                    y1 = self._cropping_coords[2]
                    y2 = self._cropping_coords[3]

                    image = image[y1:y2, x1:x2]

                ground_mask = self.old_fit_circle_to_pupil(
                    frame_num=frame_num, frame=image)['mask']
                predicted_mask = self.fit_circle_to_pupil(
                    frame_num=frame_num, frame=image)['mask']

                score_dict = compute_segmentation_score(
                    ground_mask, predicted_mask)

                score_mat[frame_num, :] = [score_dict['dice_coeff'], score_dict['jaccard_index'],
                                           score_dict['sensitivity'], score_dict['precision']]

        df_score = pd.DataFrame(score_mat, columns=[
                                'dice_coefficient', 'jaccard_index', 'sensitivity', 'precision'])
        
        return df_score

