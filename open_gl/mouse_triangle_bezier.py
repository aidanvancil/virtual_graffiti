import numpy as np
from glumpy import app, gl, gloo
from glumpy.app import clock

vertex = """
uniform vec2 resolution;
uniform float antialias;
uniform float thickness;
uniform float linelength;
attribute vec4 prev, curr, next;
varying vec2 v_uv;
void main() {
    float w = thickness/2.0 + antialias;
    vec2 p;
    if (prev.xy == curr.xy) {
        vec2 t1 = normalize(next.xy - curr.xy);
        vec2 n1 = vec2(-t1.y, t1.x);
        v_uv = vec2(-w, curr.z*w);
        p = curr.xy - w*t1 + curr.z*w*n1;
    } else if (curr.xy == next.xy) {
        vec2 t0 = normalize(curr.xy - prev.xy);
        vec2 n0 = vec2(-t0.y, t0.x);
        v_uv = vec2(linelength+w, curr.z*w);
        p = curr.xy + w*t0 + curr.z*w*n0;
    } else {
        vec2 t0 = normalize(curr.xy - prev.xy);
        vec2 t1 = normalize(next.xy - curr.xy);
        vec2 n0 = vec2(-t0.y, t0.x);
        vec2 n1 = vec2(-t1.y, t1.x);
        vec2 miter = normalize(n0 + n1);
        float dy = w / dot(miter, n1);
        v_uv = vec2(curr.w, curr.z*w);
        p = curr.xy + dy*curr.z*miter;
    }
    p.y = resolution.y - p.y;
    gl_Position = vec4(2.0*p/resolution-1.0, 0.0, 1.0);
} """

fragment = """
uniform float antialias;
uniform float thickness;
uniform float linelength;
varying vec2 v_uv;

void main() {
    float d = 0;
    float w = thickness/2.0 - antialias;

    // Cap at start
    if (v_uv.x < 0)
        d = length(v_uv) - w;

    // Cap at end
    else if (v_uv.x >= linelength)
        d = length(v_uv - vec2(linelength,0)) - w;

    // Body
    else
        d = abs(v_uv.y) - w;

    if( d < 0) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        d /= antialias;
        gl_FragColor = vec4(0.0, 0.0, 0.0, exp(-d*d));
    }
} """


# Create a window with a valid GL context

window = app.Window(860, 860, color=(1,1,1,1))
clock.set_fps_limit(15)

@window.event
def on_resize(width, height):
    quad["resolution"] = width, height
    print(width, height)

@window.event
def on_init():
    window.clear()

@window.event
def on_draw(dt):
#    window.clear()
    quad.draw(gl.GL_TRIANGLE_STRIP)

@window.event
def on_mouse_press(x, y, button):
    global line_segment
    global segment_length
    line_segment.append((x,y))
    segment_length = 1

@window.event
def on_mouse_drag(x, y, dx, dy, buttons):
    global line_segment
    global segment_length
    line_segment.append(((line_segment[segment_length-1][0]+dx), (line_segment[segment_length-1][1]+dy)))
    segment_length += 1
    if (segment_length == 4):
        line_array = np.array(line_segment, np.float32)
        P = bezier(line_array[0], line_array[1], line_array[2], line_array[3])
        V_prev, V_curr, V_next, length = bake(P)
        line_buffer1["prev"] = V_prev
        line_buffer1["curr"] = V_curr
        line_buffer1["next"] = V_next
        quad["linelength"] = length
        quad.bind(line_buffer1)
        last_point = line_segment[3]
        line_segment.clear()
        line_segment.append(last_point)
        segment_length = 1

@window.event
def on_mouse_release(x, y, button):
    global line_segment
    global segment_length
    global fill
    #line_segment.append(((line_segment[segment_length-1][0]+x), (line_segment[segment_length-1][1]+y)))
    line_segment.append((x,y))
    segment_length += 1
    if (segment_length == 4):
        line_array = np.array(line_segment, np.float32)
        P = bezier(line_array[0], line_array[1], line_array[2], line_array[3])
        V_prev, V_curr, V_next, length = bake(P)
        line_buffer1["prev"] = V_prev
        line_buffer1["curr"] = V_curr
        line_buffer1["next"] = V_next
        quad["linelength"] = length
        quad.bind(line_buffer1)
        segment_length = 0
        line_segment.clear()
    else:
        line_segment.clear()
        segment_length = 0

def bake(P, closed=False):
    epsilon = 1e-10
    n = len(P)
    if closed and ((P[0]-P[-1])**2).sum() > epsilon:
        P = np.append(P, P[0])
        P = P.reshape(n+1,2)
        n = n+1
    V = np.zeros(((1+n+1),2,4), dtype=np.float32)
    V_prev, V_curr, V_next = V[:-2], V[1:-1], V[2:]
    V_curr[...,0] = P[:,np.newaxis,0]
    V_curr[...,1] = P[:,np.newaxis,1]
    V_curr[...,2] = 1,-1
    L = np.cumsum(np.sqrt(((P[1:]-P[:-1])**2).sum(axis=-1))).reshape(n-1,1)
    V_curr[1:,:,3] = L
    if closed:
        V[0], V[-1] = V[-3], V[2]
    else:
        V[0], V[-1] = V[1], V[-2]
    return V_prev, V_curr, V_next, L[-1]

def bezier(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    t = 0
    points = []
    for x in range(0,11):
        x = ((((1-t)**3)*x1)+(3*((1-t)**2)*t*x2)+(3*((1-t)*(t**2)*x3))+((t**3)*x4))
        y = ((((1-t)**3)*y1)+(3*((1-t)**2)*t*y2)+(3*((1-t)*(t**2)*y3))+((t**3)*y4))
        points.append((x,y))
        t += (1/10)
    return np.array(points)

line_segment = []
segment_length = 0

line_buffer1 = np.zeros((11,2), [("prev", np.float32, 4), ("curr", np.float32, 4), ("next", np.float32, 4)])

line_buffer1 = line_buffer1.view(gloo.VertexBuffer)

quad = gloo.Program(vertex, fragment)
quad["antialias"] = 1.5
quad["thickness"] = 8

quad.bind(line_buffer1)

app.run()
