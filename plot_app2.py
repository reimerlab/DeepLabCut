import os
import glob
import numpy as np
import pandas as pd
import cv2

# bokeh
import bokeh
import colorcet


from bokeh.io import curdoc
from bokeh.layouts import column, gridplot, layout
from bokeh.models import ColumnDataSource, HoverTool, LabelSet
from bokeh.models.annotations import Title
from bokeh.models.widgets import TextInput, Slider
from bokeh.plotting import figure

start = 15600
end = 20000

case = '20892_9_00010_compressed_cropped_{}_{}_ellipse'.format(start, end)

data = pd.read_hdf(case + '.h5')

width = data['image'][0].shape[0]
height = data['image'][0].shape[1]

x = list(range(start, start + len(data['circle_radius'])))

source = ColumnDataSource(
    data=dict(
        image=[data['image'][0]],
        circle_radius=[data['circle_radius'][0]],
        major_radius=[data['major_radius'][0]],
        minor_radius=[data['minor_radius'][0]],
        frame_num = [x[0]],
        circle_radius_text = ["{:.2f}".format(data['circle_radius'][0])],
        major_radius_text =["{:.2f}".format(data['major_radius'][0])],
        minor_radius_text =["{:.2f}".format(data['minor_radius'][0])]
    )
)

slider = Slider(start=start, end=(
    start+data.shape[0]-1), value=start, step=1, title="Frame")

# circle fit plot
plot_line_circle = figure(width=800, height=300)
plot_line_circle.line(x=x, y=data['circle_radius'])
plot_line_circle.circle(x='frame_num', y='circle_radius', size=20, fill_color="firebrick",
                 fill_alpha=0.5, line_color=None, source=source)
labels_circle = LabelSet(x='frame_num', y='circle_radius', text='circle_radius_text', level='glyph',
              x_offset=5, y_offset=5, source=source, render_mode='canvas')
plot_line_circle.title.text = "Circle Radius"
plot_line_circle.title.text_font_size = '15pt'

plot_line_circle.add_layout(labels_circle)

# ellipse fit plot
plot_line_ellipse = figure(width=800, height=300)

# major radius
plot_line_ellipse.line(x=x, y=data['major_radius'], color='mediumpurple', legend='major_radius')
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
plot_line_ellipse.title.text = "Ellipse Radius"
plot_line_ellipse.title.text_font_size = '15pt'

plot_line_ellipse.legend.location = "top_left"
plot_line_ellipse.legend.click_policy = "hide"

plot_line_ellipse.add_layout(labels_ellipse_major)
plot_line_ellipse.add_layout(labels_ellipse_minor)

plot_fig = figure(x_range=(0, width), y_range=(0, height))

plot_fig.title.text = "Frame num: {}".format(slider.value)

plot_fig.image_rgba(image='image', x=0, y=0, dw=width, dh=height,
                    source=source)
plot_fig.axis.visible = False
plot_fig.title.text_font_size = '15pt'


def update(attr, old, new):

    # if np.nan, conver to string so that it JSON format understands
    source.data = dict(image=[data['image'][slider.value-start]],
                       circle_radius=['nan' if np.isnan(data['circle_radius'][slider.value-start]) else data['circle_radius'][slider.value-start]],
                       major_radius=['nan' if np.isnan(data['major_radius'][slider.value-start]) else data['major_radius'][slider.value-start]],
                       minor_radius=['nan' if np.isnan(data['minor_radius'][slider.value-start]) else data['minor_radius'][slider.value-start]],
                       frame_num = [x[slider.value-start]],
                       circle_radius_text = ["{:.2f}".format(data['circle_radius'][slider.value-start])],
                       major_radius_text =["{:.2f}".format(data['major_radius'][slider.value-start])],
                       minor_radius_text =["{:.2f}".format(data['minor_radius'][slider.value-start])]
                       )
    plot_fig.title.text = "Frame num: {}".format(slider.value)


slider.on_change('value', update)

layout_frame = column(gridplot([[plot_fig, column(plot_line_circle, plot_line_ellipse)]]), slider)

curdoc().add_root(layout_frame)
