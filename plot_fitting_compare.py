import os
import glob
import numpy as np
import pandas as pd
import cv2

# bokeh
import bokeh
import colorcet


from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, layout, row
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, Label
from bokeh.models.annotations import Title
from bokeh.models.widgets import TextInput, Slider
from bokeh.plotting import figure


def make_source(data):
    source = ColumnDataSource(
        data=dict(
            image=[data['image'][0]],
            circle_radius=[data['circle_radius'][0]],
            major_radius=[data['major_radius'][0]],
            minor_radius=[data['minor_radius'][0]],
            frame_num=[x[0]],
            circle_radius_text=["{:.2f}".format(data['circle_radius'][0])],
            major_radius_text=["{:.2f}".format(data['major_radius'][0])],
            minor_radius_text=["{:.2f}".format(data['minor_radius'][0])]
        )
    )
    return source


def plot_figure(source, case_num, width, height):

    plot_fig = figure(x_range=(0, width), y_range=(
        0, height), width=width, height=height)

    plot_fig.title.text = "Frame num: {}".format(slider.value)

    plot_fig.image_rgba(image='image', x=0, y=0, dw=width, dh=height,
                        source=source)
    plot_fig.axis.visible = False
    plot_fig.title.text_font_size = '15pt'

    plot_case = Label(x=10, y=250, text='Label num >= {}'.format(case), render_mode='css', border_line_color='white',
                      border_line_alpha=1.0, background_fill_color='white', background_fill_alpha=1.0)

    plot_fig.add_layout(plot_case)

    return plot_fig


def plot_circle_radius(data, source):
    plot_line_circle = figure(width=800, height=300)
    plot_line_circle.line(x=x, y=data['circle_radius'])
    plot_line_circle.circle(x='frame_num', y='circle_radius', size=20, fill_color="firebrick",
                            fill_alpha=0.5, line_color=None, source=source)
    labels_circle = LabelSet(x='frame_num', y='circle_radius', text='circle_radius_text', level='glyph',
                             x_offset=5, y_offset=5, source=source, render_mode='canvas')
    plot_line_circle.title.text = "Circle Radius"
    plot_line_circle.title.text_font_size = '15pt'

    plot_line_circle.add_layout(labels_circle)

    return plot_line_circle


def plot_ellipse_radius(data, source, case_num):
    plot_line_ellipse = figure(width=800, height=300)

    # major radius
    plot_line_ellipse.line(
        x=x, y=data['major_radius'], color='mediumpurple', legend='major_radius')
    plot_line_ellipse.circle(x='frame_num', y='major_radius', size=20, fill_color="firebrick",
                             fill_alpha=0.5, line_color=None, source=source)
    labels_ellipse_major = LabelSet(x='frame_num', y='major_radius', text='major_radius_text', level='glyph',
                                    x_offset=5, y_offset=5, source=source, render_mode='canvas')

    # minor radius
    plot_line_ellipse.line(x=x, y=data['minor_radius'], legend='minor_radius')
    plot_line_ellipse.circle(x='frame_num', y='minor_radius', size=20, fill_color="firebrick",
                             fill_alpha=0.5, line_color=None, source=source)
    labels_ellipse_minor = LabelSet(x='frame_num', y='minor_radius', text='minor_radius_text', level='glyph',
                                    x_offset=5, y_offset=5, source=source, render_mode='canvas')
    plot_line_ellipse.title.text = "Ellipse Radius Label num >= {}".format(
        case_num)
    plot_line_ellipse.title.text_font_size = '15pt'

    plot_line_ellipse.legend.location = "top_left"
    plot_line_ellipse.legend.click_policy = "hide"

    plot_line_ellipse.add_layout(labels_ellipse_major)
    plot_line_ellipse.add_layout(labels_ellipse_minor)

    return plot_line_ellipse


start = 15600
end = 20000

cases = [5, 6, 7, 8]

data_dict = {}
for i in cases:
    case = '20892_9_00010_compressed_cropped_{}_{}_ellipse_{}'.format(
        start, end, i)
    data_dict['data{}'.format(i)] = pd.read_hdf(case + '.h5')

width = data_dict['data5']['image'][0].shape[0]
height = data_dict['data5']['image'][0].shape[1]

x = list(range(start, start + len(data_dict['data5']['circle_radius'])))

# make a slider
slider = Slider(start=start, end=(
    start+len(x)-1), value=start, step=1, title="Frame")

source_dict = {}
figure_dict = {}
ellipse_dict = {}

for case in cases:
    source_dict['source{}'.format(case)] = make_source(
        data_dict['data{}'.format(case)])
    figure_dict['figure{}'.format(case)] = plot_figure(
        source_dict['source{}'.format(case)], case_num=case, width=width + 100, height=height + 100)
    ellipse_dict['ellipse{}'.format(case)] = plot_ellipse_radius(
        data=data_dict['data{}'.format(case)], source=source_dict['source{}'.format(case)], case_num=case)

# circle line plot
circle_line_plot = plot_circle_radius(
    data=data_dict['data5'], source=source_dict['source5'])


def update(attr, old, new):

    for case in cases:
        source_dict['source{}'.format(case)].data = dict(image=[data_dict['data{}'.format(case)]['image'][slider.value-start]],
                                                         circle_radius=['nan' if np.isnan(
                                                             data_dict['data{}'.format(case)]['circle_radius'][slider.value-start]) else data_dict['data{}'.format(case)]['circle_radius'][slider.value-start]],
                                                         major_radius=['nan' if np.isnan(
                                                             data_dict['data{}'.format(case)]['major_radius'][slider.value-start]) else data_dict['data{}'.format(case)]['major_radius'][slider.value-start]],
                                                         minor_radius=['nan' if np.isnan(
                                                             data_dict['data{}'.format(case)]['minor_radius'][slider.value-start]) else data_dict['data{}'.format(case)]['minor_radius'][slider.value-start]],
                                                         frame_num=[
                                                             x[slider.value-start]],
                                                         circle_radius_text=["{:.2f}".format(
                                                             data_dict['data{}'.format(case)]['circle_radius'][slider.value-start])],
                                                         major_radius_text=["{:.2f}".format(
                                                             data_dict['data{}'.format(case)]['major_radius'][slider.value-start])],
                                                         minor_radius_text=["{:.2f}".format(
                                                             data_dict['data{}'.format(case)]['minor_radius'][slider.value-start])]
                                                         )
        figure_dict['figure{}'.format(
            case)].title.text = "Frame num: {}".format(slider.value)


slider.on_change('value', update)

figure_grid = column(figure_dict['figure5'], figure_dict['figure6'],
                     figure_dict['figure7'], figure_dict['figure8'])

plot_column = column(circle_line_plot, ellipse_dict['ellipse5'],
                     ellipse_dict['ellipse6'], ellipse_dict['ellipse7'], ellipse_dict['ellipse8'])

layout_frame = column(
    row(figure_grid, plot_column),
    slider

)

curdoc().add_root(layout_frame)
