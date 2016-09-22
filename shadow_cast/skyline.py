import numpy as np
import preperation.ascii as ascii
import matplotlib.pyplot as plt

def skyline(camera_pos,path):

    # hillshade
    _, hillshade = ascii.read_ascii("C:\Arbeitsprojekte\Wasim\Gebiete\Brixental\Final_Szenarios\generel/hillshade.asc")

    def angle_between(point, target, clockwise=True):
        angle = np.rad2deg(np.arctan2(target[1] - point[1], target[0] - point[0]))
        if not clockwise:
            angle = -angle
        return angle % 360

    headinfo, z = ascii.read_ascii(path)

    x0 = (camera_pos[0] - headinfo[2])/headinfo[4]
    y0 = ((headinfo[3]+headinfo[1]*headinfo[4])-camera_pos[1])/headinfo[4]

    # outer bound pixel of raster, index
    xv, yv = np.meshgrid(np.arange(z.shape[0]), np.arange(z.shape[1]))
    x1 = np.concatenate([xv[y0:, -1], np.flipud(xv[-1, :-1]), np.flipud(xv[:-1, 0]), xv[0, 1:], xv[1:y0, -1]])
    y1 = np.concatenate([yv[y0:, -1], np.flipud(yv[-1, :-1]), np.flipud(yv[:-1, 0]), yv[0, 1:], yv[1:y0, -1]])

    # Create lines clockwise, starting in right direction
    point = zip(x1, y1)
    length = [int(np.hypot(p[0] - x0, p[1] - y0)) for p in point]
    x = [np.linspace(x0, p[0], length[i]) for i, p in enumerate(point)]
    y = [np.linspace(y0, p[1], length[i]) for i, p in enumerate(point)]
    x, y = [x.astype(int) for x in x], [y.astype(int) for y in y]

    # Extract the values along the line
    pairs = zip(x, y)
    zi = [z[pairs[i]] for i, _ in enumerate(pairs)]

    # calculate important components
    dist = np.sqrt(np.power((x0 - x1), 2) + np.power((y0 - y1), 2))
    height_difference = zi - z[y0, x0]
    angle = [np.rad2deg(np.arctan(height_difference[i] / dist[i])) for i, _ in enumerate(height_difference)]
    direction = angle_between((x0, y0), (x1, y1))
    max_angle = [np.max(angle[i]) for i, _ in enumerate(angle)]
    max_angle_index = [np.where(angle[i] == np.max(angle[i]))[0][0] for i, _ in enumerate(angle)]

    # Skyline Position in raster
    x_pos = [x[i][max_angle_index[i]] for i, _ in enumerate(max_angle_index)]
    y_pos = [y[i][max_angle_index[i]] for i, _ in enumerate(max_angle_index)]


    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow(z, cmap='Greys')
    axes[0,0].plot(x_pos, y_pos, 'r-')
    axes[0,0].plot(x0, y0, 'bo')
    axes[0,0].axis('image')
    #
    # axes[1].plot(zi)

    axes[1,0].plot(direction, max_angle, "r-")
    # axes[1].plot(direction+360,max_angle, "r-")

    plt.show()





camera_pos = (288627,5256213)
skyline(camera_pos, "C:\Arbeitsprojekte\Wasim\Gebiete\Brixental\Final_Szenarios\generel/dgm_50.asc")

