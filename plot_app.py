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

start = 6000
end = 7200

case = '20892_9_00010_compressed_cropped_{}_{}'.format(start, end)

data = pd.read_hdf(case + '.h5')

width = data['image'][0].shape[0]
height = data['image'][0].shape[1]

x = list(range(6900, 6900 + len(data['radius'])))

source = ColumnDataSource(
    data=dict(
        image=[data['image'][0]],
        radius=[data['radius'][0]],
        frame_num = [x[0]],
        radius_text =["{:.2f}".format(data['radius'][0])]
    )
)

slider = Slider(start=start, end=(
    start+data.shape[0]-1), value=start, step=1, title="Frame")


plot_line = figure(width=800, height=600)
plot_line.line(x=x, y=data['radius'])
plot_line.circle(x='frame_num', y='radius', size=20, fill_color="firebrick",
                 fill_alpha=0.5, line_color=None, source=source)
labels = LabelSet(x='frame_num', y='radius', text='radius_text', level='glyph',
              x_offset=5, y_offset=5, source=source, render_mode='canvas')

plot_line.add_layout(labels)

plot_fig = figure(x_range=(0, width), y_range=(0, height))

plot_fig.title.text = "Frame num: {}".format(slider.value)

plot_fig.image_rgba(image='image', x=0, y=0, dw=width, dh=height,
                    source=source)
plot_fig.axis.visible = False
plot_fig.title.text_font_size = '15pt'


def update(attr, old, new):
    source.data = dict(image=[data['image'][slider.value-start]],
                       radius=[data['radius'][slider.value-start]],
                       frame_num = [x[slider.value-start]],
                       radius_text =["{:.2f}".format(data['radius'][slider.value-start])]
                       )
    plot_fig.title.text = "Frame num: {}".format(slider.value)


slider.on_change('value', update)

layout_frame = layout([
    [plot_fig, plot_line],
    [slider]
], sizeing_mode='stretch_both')

curdoc().add_root(layout_frame)
