import random

# creates a box with screen size
# randomize function makes it randomly smaller within screen
class Box():
    def __init__(self, screen):
        self.screen = screen
        self.x = 0
        self.y = 0
        self.xlen = screen[0]
        self.ylen = screen[1]

    def randomize(self):
        self.x = random.randint(0, self.screen[0])
        self.y = random.randint(0, self.screen[1])
        self.xlen = random.randint(0, self.screen[0] - self.x)
        self.ylen = random.randint(0, self.screen[1] - self.y)


# generate boxes on screen
def init_scene(screen, box_count=5):
    boxes = []
    for i in range(box_count):
        box = Box(screen)
        box.randomize()
        boxes.append(box)

    return boxes
