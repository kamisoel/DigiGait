import plotly.graph_objs as go
import plotly.express as px

import pandas as pd
import numpy as np

def frame_args(duration, redraw=True, transition=False):
    return {
            "frame": {"duration": duration, "redraw": redraw},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": 0},
        }

def create_skeleton_fig(pose_3d, skeleton=None, joints=None, 
                        eye=None, fps=50, height=500):
    if skeleton is None:
        skeleton = [-1,  0,  1,  2,  0,  4,  5,  0, \
                     7,  8,  9,  8, 11, 12,  8, 14, 15]
    if joints is None:
        joints = "MHip, RHip, RKnee, RAnkle, LHip, LKnee, \
                 LAnkle, Spine, Neck, Nose, Head, LShoulder, \
                 LElbow, LWrist, RShoulder, RElbow, RWrist".split(", ")

    if eye is None:
        eye = dict(x=-1.0, y=3.0, z=.5)
        
    lines = {'frame': [], 'joint': [], 'x':[], 'y':[], 'z':[]}
    for f in range(len(pose_3d)):
        for j in range(len(joints)):
            p = skeleton[j]
            if p != -1:
                lines['frame'].extend([f]*3)
                lines['joint'].extend([joints[j], joints[p], None])
                for i, c in enumerate(list('xyz')):
                    lines[c].append(pose_3d[f, j, i])
                    lines[c].append(pose_3d[f, p, i])
                    lines[c].append(None)
    pose_df = pd.DataFrame.from_dict(lines)
    
    # Create figure
    frames = [go.Frame(
            name=str(frame),
            data=[go.Scatter3d(x=df['x'], y=df['y'], z=df['z'],
                    mode='markers+lines', line=dict(width=5),
                    marker=dict(size=5),
                    hovertemplate= '<b>%{text}</b><br>'+
                                   '<b>x</b>: %{x:.3f}<br>'+
                                   '<b>y</b>: %{y:.3f}<br>'+
                                   '<b>z</b>: %{z:.3f}<br>'+
                                   '<extra></extra>',
                    text = df['joint']
            )])
            for frame, df in pose_df.groupby('frame')]
    
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 15},
            "prefix": "Frame:",
            "xanchor": "right"
        },
        "pad": {"b": 10, "t": 15},
        "len": 0.7,
        "x": 0.25,
        "y": 0,
        "steps": [{
            "args": [
                [frame], frame_args(0)
            ],
            "label": frame,
            "method": "animate"}
        for frame in range(0, len(pose_3d)+1, fps)]
    }

    layout=go.Layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)', # transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, b=0, t=0), # tight layout
        scene = go.layout.Scene( # scene dimension
            xaxis=dict(range=[-.75,.75], autorange=False, zeroline=False),
            yaxis=dict(range=[-.75,.75], autorange=False, zeroline=False),
            zaxis=dict(range=[-0.2, 2], autorange=False, zeroline=False),
            aspectratio=dict(x=1, y=1, z=2.),
        ),
        scene_camera=dict(
            eye=eye,
        ),
        hovermode="closest",
        height=height, #width=400,
        sliders=[sliders_dict],
        updatemenus=[{
            "buttons":[{
                        "args": [None, frame_args(1./fps)],
                        "label": "&#9654;", # play symbol
                        "method": "animate"
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate"
                    }],
            "direction": "left",
            "pad": {"r": 10, "t": 40},
            "showactive": False,
            "type": "buttons",
            "x": 0,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top"
        }]
    )

    return go.Figure(data=frames[0].data, layout=layout, frames=frames)

def create_angle_figure(angles, gait_cycles=[], joint='Knee'):
    names = ['Right '+joint, 'Left '+joint]
    fig = go.Figure()#make_subplots(2, 1, shared_xaxes=True)
    for i in range(len(names)):
        fig.add_trace(
            go.Scatter(
                y=angles[:,i],
                name=names[i], meta=names[i],
                hovertemplate= '%{meta}: %{y:.1f}°'+
                                '<extra></extra>'
            )#, i+1, 1
        )
    #fig.update_yaxes(matches='y')
    fig.update_layout(
        dragmode= 'pan', 
        xaxis=dict(range=[0,300], title='Frame'), 
        yaxis=dict(fixedrange=True, title='Knee Extension/Flexion'),
        margin=dict(l=10, r=10, b=10, t=10),
        hovermode="x unified",
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        hoverlabel_bgcolor='black',
    )
    y_min, y_max = (-25, 90)
    fig.add_shape(
        dict(type="line", x0=0, x1=0, y0=y_min, y1=y_max, line_color="green"), 
        #row="all", col=1
    )
    for x in gait_cycles:
        fig.add_shape(
            dict(type="line", x0=x, x1=x, y0=y_min, y1=y_max, line_color="orange")
        )
    return fig


def create_gait_phase_figure(angles, norm_data=None, joint='Knee'):
    names = ['Right '+joint, 'Left '+joint]
    norm_color = 'rgba(162,162,162,0.5)'
    fig = go.Figure()

    if norm_data is not None:
        mean, std = norm_data
        min_norm = mean - 2 * std
        max_norm = mean + 2 * std
        x = np.arange(len(mean))
        fig.add_trace(
            go.Scatter( x=np.concatenate([x, x[::-1]]), 
                        y=np.concatenate([max_norm, min_norm[::-1]]), 
                        fill='tozerox', showlegend=False, mode='none',
                        hoverinfo='skip', legendgroup='Norm', fillcolor=norm_color)
        )
        fig.add_trace(
            go.Scatter(y=mean, name='Norm value', meta='Norm value',
                        legendgroup='Norm', line_color=norm_color,
                        hovertemplate= '%{meta}: %{y:.1f}°<extra></extra>')),
    for i in range(len(names)):
        fig.add_trace(
            go.Scatter(
                y=angles[i], name=names[i], meta=names[i],
                hovertemplate= '%{meta}: %{y:.1f}°<extra></extra>'
            )
        )
    fig.update_layout(
        dragmode= 'pan', 
        xaxis=dict(range=[0,100], fixedrange=True, title='% Gait Cycle'), 
        yaxis=dict(fixedrange=True, title='Avg. Knee Extension/Flexion'),
        margin=dict(l=10, r=10, b=10, t=10),
        hovermode="x unified",
        template='plotly_dark',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        hoverlabel_bgcolor='black',
    )
    return fig