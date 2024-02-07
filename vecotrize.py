import numpy as np

# find intersects for perpendicular segments
def dumb_intersect(a,b):
    da = a[1]-a[0]
    db = b[1]-b[0]

    # check parallel
    if np.all(np.dot(da,db) == 0):

        # check if between TODO
        return a[0]*(a[0]==a[1])+b[0]*(b[0]==b[1])

    return None


# find all block intersects
def overlap_intersects(segments):
    points = []

    for i in range(len(segments)):
        for j in range(i+1,len(segments)):
            a = np.asarray(segments[i])
            b = np.asarray(segments[j])
            test = np.append(a, b, axis=0)
            if len(np.unique(test, axis=0))<4:
                continue
            intersect = dumb_intersect(a,b)
            if intersect is None:
                continue

            if not any((any(np.isnan(intersect)), any(np.isinf(intersect)))):
                points.append(intersect)
    points = np.asarray(points)

    return points


# splitting all boxes into points and segments
def points_from_segments(segments):
    points = segments.reshape(-1, 2)
    return np.unique(points, axis=0)


def segments_from_box(box):
    x1, y1 = box.x, box.y
    x2, y2 = box.x+box.xlen, box.y+box.ylen
    s = []
    s.append([[x1,y1],[x2,y1]])
    s.append([[x2,y1],[x2,y2]])
    s.append([[x2,y2],[x1,y2]])
    s.append([[x1,y2],[x1,y1]])
    return s


def segments_from_boxes(box_list):
    segments = []
    for box in box_list:
        s = segments_from_box(box)
        segments.extend(s)
    return segments