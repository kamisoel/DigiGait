import c3d
import math
import numpy as np
from data.timeseries_utils import interp_along_time

def write_c3d(handle, pose, angle, events, freq=50):
    writer = c3d.Writer(point_rate=freq, analog_rate=freq)
    
    writer.set_point_labels(['MHIP', 'RHIP', 'RKNEE', 'RHEEL',
                            'LHIP', 'LKNEE', 'LHEEL', 'SPINE',
                            'THORAX', 'NECK', 'HEAD', 'LSHOULDER',
                            'LELBOW', 'LWRIST', 'RHOULDER', 'RELBOW',
                            'RWRIST'])
    writer.set_analog_labels(['RKneeAngleZ', 'LKneeAngleZ'])

    # save up to 18 events in the header, e.g 4 full cycles
    event_pairs = [] # list of pairs (time:float, label:str)
    for i, label in enumerate(['RHS', 'LHS', 'RTO', 'TLO']):  
        for e in events[i][:4]:   # up to 4 cycles
            event_pairs.append((e / freq, label)) # save e as time in seconds
    writer.header.encode_events(event_pairs)

    for i in range(len(pose)):
        p = np.pad(pose[i], [(0,0),(0,2)]) # (17, 5)
        a = angle[i][np.newaxis, :, np.newaxis] # (1, 2, 1)
        writer.add_frames((p, a))

    writer.write(handle)

def read_c3d(handle):
    reader = c3d.Reader(handle)
    if len(reader.point_labels) != 17:
        raise ValueError("Wrong keypoint topology!")

    events = [[] for _ in range(4)] # list of 4 arrays with event timings in frames
    for event, label in reader.header.events:
        i = ['RHS', 'LHS', 'RTO', 'TLO'].index(label[:3]) # look up index of first three letters
        events[i].append(round(event * reader.point_rate)) #append frame number in event group
    events = [np.array(e, dtype=np.int) for e in events] #convert to np.ndarray

    p = np.zeros((reader.frame_count, 17, 3))
    a = None

    if reader.analog_used >= 2:
        analog_labels = reader.analog_labels.tolist()
        if "RKneeAngleZ" in analog_labels and "LKneeAngleZ" in analog_labels:
            a = np.zeros((reader.frame_count, 2))
    for i, points, analog in reader.read_frames():
        p[i-1] = points[:, :3]
        if a is not None: # analog data in file
            indices = [analog_labels.index('RKneeAngleZ'), 
                       analog_labels.index('LKneeAngleZ')]
            a[i-1] = analog[indices, 0]
    
    if not math.isclose(reader.point_rate, 50):
        pass # interpolate
    return p, a, events
