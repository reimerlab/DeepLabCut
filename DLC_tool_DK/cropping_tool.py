"""
DeepLabCut Toolbox
D Kim, donniek@bcm.edu

***DISCLAIMER***
The code below is the product of adaption of matplotlib example:
https://matplotlib.org/examples/event_handling/viewlims.html

You can use it anyway you want/modify however you want on your behalf.
"""
import os
import yaml
import cv2
from pathlib import Path
import ruamel.yaml

from deeplabcut.utils import auxiliaryfunctions

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# We just subclass Rectangle so that it can be called with an Axes
# instance, causing the rectangle to update its shape to match the
# bounds of the Axes


class UpdatingRect(Rectangle):
    def __call__(self, ax):
        self.set_bounds(*ax.viewLim.bounds)
        ax.figure.canvas.draw_idle()


class ZoomedDisplay():
    """
    Display the zoomed in area in the left panel.
    """

    def __init__(self, frame, height, width):
        self.frame = frame
        # in pixels
        self.xstart = 0
        self.xend = width
        self.ystart = 0
        self.yend = height

    def ax_update(self, ax):
        ax.set_autoscale_on(False)  # Otherwise, infinite loop

        # Get the range for the new area
        # viewLim.bounds give leftbottom and righttop coordinates
        xstart, yend, xdelta, ydelta = ax.viewLim.bounds
        xend = xstart + xdelta
        ystart = yend + ydelta

        self.xstart = round(xstart).astype(int)
        self.xend = round(xend).astype(int)
        self.ystart = round(ystart).astype(int)
        self.yend = round(yend).astype(int)

        # images are in row major order. user should double check if major order
        bounded_frame = self.frame[self.ystart:self.yend,
                                   self.xstart:self.xend]
        im = ax.images[-1]
        im.set_data(bounded_frame)
        # extent is data axes (left, right, bottom, top) for making image plots registered with data plots.
        # https://matplotlib.org/api/image_api.html#matplotlib.image.AxesImage.set_extent
        im.set_extent((xstart, xend, yend, ystart))
        ax.figure.canvas.draw_idle()


def update_config_crop_coords(config):
    """
    Given the list of videos, users can manually zoom in the area they want to crop and update the coordinates in config.yaml 

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.
    """
    config_file = Path(config).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)

    video_sets = cfg['video_sets'].keys()
    for vindex, video_path in enumerate(video_sets):

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(
                cv2.CAP_PROP_FRAME_WIDTH))

            print("video {}: {} has original dim in {} by {}".format(
                vindex, video_path, width, height))

            # putting the frame to read at the very middle of the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((nframes-1)/2))
            res, frame = cap.read()

            display = ZoomedDisplay(frame=frame, height=height, width=width)

            fig1, (ax1, ax2) = plt.subplots(1, 2)

            ax1.imshow(frame)
            ax2.imshow(frame)

            rect = UpdatingRect([0, 0], 0, 0, facecolor='None',
                                edgecolor='red', linewidth=1.0)
            rect.set_bounds(*ax2.viewLim.bounds)
            ax1.add_patch(rect)

            # Connect for changing the view limits
            ax2.callbacks.connect('xlim_changed', rect)
            ax2.callbacks.connect('ylim_changed', rect)

            ax2.callbacks.connect('xlim_changed', display.ax_update)
            ax2.callbacks.connect('ylim_changed', display.ax_update)
            ax2.set_title("Zoom here")

            plt.show()

            new_width = display.xend - display.xstart
            new_height = display.yend - display.ystart

            print("your cropped coords are {} {} {} {} with dim of {} by {} \n".format(
                display.xstart, display.xend, display.ystart, display.yend, new_width, new_height))

            cfg['video_sets'][video_path] = {'crop': ', '.join(
                map(str, [display.xstart, display.xend, display.ystart, display.yend]))}

            cap.release()
            plt.close("all")

        else:
            print("Cannot open the video file: {} !".format(video_path))

    # Update the yaml config file
    auxiliaryfunctions.write_config(config, cfg)


def update_inference_cropping_config(cropping_config, video_path):
    """
    Given a video path, users can manually zoom in the area they want to crop and update cropping coordinates in config.yaml 

    Parameters
    ----------
    cropping_config: string
        Full path of the inference_cropping.yaml file as a string.
    video_path : str
        Full path to the video

    Output:
        new cropping coordinates: list
            list contatining xstart, xend, ystart, and yend
    """
    config_file = Path(cropping_config).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(
            cv2.CAP_PROP_FRAME_WIDTH))

        print("original dim in {} by {}".format(width, height))

        # putting the frame to read at the very middle of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((nframes-1)/2))
        _, frame = cap.read()

        display = ZoomedDisplay(frame=frame, height=height, width=width)

        fig1, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(frame)
        ax2.imshow(frame)

        rect = UpdatingRect([0, 0], 0, 0, facecolor='None',
                            edgecolor='red', linewidth=1.0)
        rect.set_bounds(*ax2.viewLim.bounds)
        ax1.add_patch(rect)

        # Connect for changing the view limits
        ax2.callbacks.connect('xlim_changed', rect)
        ax2.callbacks.connect('ylim_changed', rect)

        ax2.callbacks.connect('xlim_changed', display.ax_update)
        ax2.callbacks.connect('ylim_changed', display.ax_update)
        ax2.set_title("Zoom here")

        plt.show()

        new_width = display.xend - display.xstart
        new_height = display.yend - display.ystart

        print("your cropped coords are {} {} {} {} with dim of {} by {} \n".format(
            display.xstart, display.xend, display.ystart, display.yend, new_width, new_height))

        cfg[video_path]['x1'] = int(display.xstart)
        cfg[video_path]['x2'] = int(display.xend)
        cfg[video_path]['y1'] = int(display.ystart)
        cfg[video_path]['y2'] = int(display.yend)

        cap.release()
        plt.close("all")

        # Update the yaml config file
        yaml = ruamel.yaml.YAML()

        yaml.dump(cfg,config_file)

        return [int(display.xstart), int(display.xend), int(display.ystart), int(display.yend)]

    else:
        print("Cannot open the video file: {} !".format(video_path))


