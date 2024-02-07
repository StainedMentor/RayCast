from pyray import *

from Scene import *
from vecotrize import *
from Cuda_Shaders import get_light_points

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 850

screen = [SCREEN_WIDTH, SCREEN_HEIGHT]

init_window(SCREEN_WIDTH, SCREEN_HEIGHT, "RAYS")

boxes = init_scene(screen)
screen_box = Box(screen)

segments = []
segments.extend(segments_from_box(screen_box))
box_segments = segments_from_boxes(boxes)
segments.extend(box_segments)
segments = np.array(segments)

points = points_from_segments(segments)
overlaps = np.asarray(overlap_intersects(box_segments))
# points = np.append(points, overlaps, axis=0)
mouse_position = (200, 200)

while not window_should_close():
    begin_drawing()
    custom_color_bg = Color(35, 42, 54,255)

    clear_background(custom_color_bg)

    mouse_position = (get_mouse_x(), get_mouse_y())

    point_mask = get_light_points(mouse_position,segments,points)
    if not np.isnan(point_mask).any():
        for i, intersect in enumerate(point_mask):

            if i < len(point_mask) - 1:
                draw_triangle(mouse_position, point_mask[i].tolist(), point_mask[i + 1].tolist(), Color(255, 255, 255, 255))

            if i == len(point_mask) - 1:
                draw_triangle(mouse_position, point_mask[i].tolist(), point_mask[0].tolist(), Color(255, 255, 255, 255))

        for intersect in point_mask:
            draw_line(mouse_position[0], mouse_position[1], int(intersect[0]), int(intersect[1]), Color(104, 27, 143,255))

    for box in boxes:
        draw_rectangle_lines(box.x, box.y, box.xlen, box.ylen, get_color(255))
    end_drawing()
close_window()
