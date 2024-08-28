import os
def del_file(path_data):
    """
    Delete folder
    Args:
        path_data:Folder path
    """
    for i in os.listdir(path_data):
        file_data = path_data + "/" + i
        if os.path.isfile(file_data) == True:
            os.remove(file_data)
        else:
            del_file(file_data)

def uv2xyz(camera_inter, scalingFactor, depth, uv):
    """
    Args:
        camera_inter: camera inter parameter
        scalingFactor: camera scalingFactor
        depth: depth image
        uv: Pixel coordinate

    Returns:
        [x,y,z]
    """
    fx, fy, centerX, centerY = camera_inter
    # -------
    # u，v
    # -------
    u, v = uv
    # ---------
    # X，Y，Z
    # ---------
    Z = depth.getpixel((u, v)) / scalingFactor
    if Z == 0 :
        return False
    else:
        X = (u - centerX) * Z / fx
        Y = (v - centerY) * Z / fy
        return [X, Y, Z]