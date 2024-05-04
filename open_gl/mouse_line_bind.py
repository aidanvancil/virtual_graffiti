import numpy as np
from glumpy import app, gl, gloo
from glumpy.app import clock
from curves import curve4_bezier

# I want to create this file as a way to test lets just say drawing a square... with

# First lets create our vertex and fragments shaders

vertex = """
    attribute vec2 position;
    attribute vec4 color;
    varying vec4 v_color;
    void main(){
        float x, y;
        x = ((position.x - 430)/430);
        y = ((430 - position.y)/430);
        gl_Position = vec4(x, y, 0.0, 1.0);
        v_color = color;
    } """

fragment = """
    varying vec4 v_color;
    void main() { gl_FragColor = v_color; } """

# Create a window with a valid GL context
window = app.Window(860, 860, color=(1,1,1,1))
clock.set_fps_limit(15)
@window.event
def on_draw(dt):
#    window.clear()
    quad.draw(gl.GL_LINE_STRIP)

@window.event 
def on_mouse_press(x, y, button):
    global line_segment
    global length
    line_segment.append((x,y))
    length = 1

@window.event
def on_mouse_drag(x, y, dx, dy, buttons):
    global line_segment
    global length
    global fill
    line_segment.append(((line_segment[length-1][0]+dx), (line_segment[length-1][1]+dy)))
    length += 1
    if (length == 4):
        line_array = np.array(line_segment, np.float32)
        if (fill == "line_buffer1"):
            line_buffer1["position"] = bezier(line_array[0], line_array[1], line_array[2], line_array[3])
            quad.bind(line_buffer1)
            last_point = line_segment[3]
            line_segment.clear()
            line_segment.append(last_point)
            length = 1
            fill = "line_buffer2"
        else:
            line_buffer2["position"] = bezier(line_array[0], line_array[1], line_array[2], line_array[3])
            quad.bind(line_buffer2)
            last_point = line_segment[3]
            line_segment.clear()
            line_segment.append(last_point)
            length = 1

            fill = "line_buffer1"

@window.event
def on_mouse_release(x, y, button):
    global line_segment
    global length
    global fill
    line_segment.append(((line_segment[x]), (line_segment[y])))
    length += 1
    if (length == 4):
        line_array = np.array(line_segment, np.float32)
        if (fill == "line_buffer1"):
            line_buffer1["position"] = bezier(line_array[0], line_array[1], line_array[2], line_array[3])
            quad.bind(line_buffer1)
            length = 0
            line_segment.clear()
            fill = "line_buffer2"
        else:
            line_buffer2["position"] = bezier(line_array[0], line_array[1], line_array[2], line_array[3])
            quad.bind(line_buffer2)
            length = 0
            line_segment.clear()
            fill = "line_buffer1"
    else:
        line_segment.clear()
        length = 0


def bezier(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    t = 0
    points = []
    for x in range(0,21):
        x = ((((1-t)**3)*x1)+(3*((1-t)**2)*t*x2)+(3*((1-t)*(t**2)*x3))+((t**3)*x4))
        y = ((((1-t)**3)*y1)+(3*((1-t)**2)*t*y2)+(3*((1-t)*(t**2)*y3))+((t**3)*y4))
        points.append((x,y))
        t += (1/20)
    return np.array(points)

# this will be used to capture the coordinates

line_segment = []
length = 0

line_buffer1 = np.zeros(21, [("position", np.float32, 2), ("color", np.float32, 4)])
line_buffer1["color"][:] = [[1,0,0,1]]

line_buffer2 = np.zeros(21, [("position", np.float32, 2), ("color", np.float32, 4)])
line_buffer2 ["color"][:] = [[1,0,0,1]]

line_buffer1 = line_buffer1.view(gloo.VertexBuffer)
line_buffer2 = line_buffer2.view(gloo.VertexBuffer)

quad = gloo.Program(vertex, fragment)
quad.bind(line_buffer2)
fill = "line_buffer1"


app.run()
