import numpy as np

GAUGE = 1435 #mm
SELECTED_LONGITUD = 250000 #mm


normal = {
    "SOURCE":np.array([
        [1125.0, 3040.0],
        [3025.0, 3040.0],
        [2380.0, 1500.0],
        [1820.0, 1500.0],
    ]),
    "DEST":np.array([
        [-GAUGE/2, 0],
        [GAUGE/2, 0],
        [GAUGE/2, SELECTED_LONGITUD-1],
        [-GAUGE/2, SELECTED_LONGITUD-1],
    ])
}
SELECTED_LONGITUD = 150000 #mm
wayside = {
    "SOURCE":np.array([
        [0.0, 585.0],
        [25.0, 640.0],
        [1260.0, 325.0],
        [1250.0, 310.0],
    ]),
    "DEST":np.array([
        [-GAUGE/2, 0],
        [GAUGE/2, 0],
        [GAUGE/2, SELECTED_LONGITUD-1],
        [-GAUGE/2, SELECTED_LONGITUD-1],
    ])
}



