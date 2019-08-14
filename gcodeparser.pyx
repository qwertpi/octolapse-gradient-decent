cdef get_points(line):
    cdef double x = float(line.split("X")[1].split(" ")[0])
    cdef double y = float(line.split("Y")[1].split(" ")[0])
    return [x, y]

def GcodeParser(filename):
    lines = [line for line in open(filename).read().split("\n")]
    valid_points = []
    sublist = []
    cdef int num_layers = 0
    for line in lines:
        if ";LAYER:" in line:
            num_layers += 1
            if num_layers != 1:
                valid_points.append(sublist)
            sublist = []
        elif "G1" in line:
            if "X" in line and "Y" in line and "E" in line:
                sublist.append(get_points(line))

    valid_points.append(sublist)
    
    return num_layers, valid_points
