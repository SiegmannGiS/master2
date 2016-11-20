import matplotlib
import matplotlib.pyplot as plt
import matplotlib_styles
import numpy as np
from PIL import Image, ImageChops
from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import gdal, gdalconst


try:
    from tvtk.api import tvtk
    from tvtk.tools import ivtk
    from tvtk.common import configure_input
    import vtkutils
except ImportError as err:
    print('Import error: {0}'.format(err))
except ValueError as err:
    print('Value error: {0}'.format(err))


def read_gdal_file(filename):
    ds = gdal.Open(filename, gdalconst.GA_ReadOnly)
    rb = ds.GetRasterBand(1)
    data = rb.ReadAsArray()
    nodata = rb.GetNoDataValue()
    if data.dtype.name.startswith('float'): data[data == nodata] = 'nan'

    geo_transform = ds.GetGeoTransform()
    num_cols = ds.RasterXSize
    num_rows = ds.RasterYSize
    cellsize = geo_transform[1]
    x0 = geo_transform[0]
    y1 = geo_transform[3] # caution, this is y1!
    x1 = x0 + num_cols * cellsize
    y0 = y1 - num_rows * cellsize
    extent = (x0, x1, y0, y1)

    return data, extent

def resize_image(image, width, height):
    image = Image.fromarray(image)
    image = image.resize((width, height), Image.NEAREST)
    return np.asarray(image)

def blend_images(img1, img2, alpha):
    """ Blends img2 on top of img1 with the specified alpha. """

    assert img1.shape == img2.shape
    assert np.isscalar(alpha) or alpha.shape == img1.shape[0:2]

    img = np.zeros(img1.shape)
    for i in range(3):
        img[:, :, i] = img1[:, :, i] * (1 - alpha) + img2[:, :, i] * alpha

    return np.cast[np.uint8](img)

def mapshow(X,
        ax=None,
        extent=None,
        figsize=None,
        background=None,
        background_cmap=matplotlib.cm.gray,
        background_min_range=None,
        background_max_range=None,
        alpha=1.,
        pdf_transparency_bugfix=False,
        cmap=None,
        colorbar='horizontal',
        colorbar_label='',
        colorbar_location=None,
        colorbar_size='8%',
        colorbar_pad=0.8 / 2.54,
        colorbar_ticklabel_format='%.0f',
        colorbar_shift_ticks=False,
        colorbar_hide_ticklines=False,
        colorbar_rasterize=False,
        colorbar_extend='neither',
        colorbar_extendfrac=None,
        discrete=False,
        dpi=None,
        interpolation=None,
        min_range=None,
        max_range=None,
        num_classes=None,
        class_boundaries=None,
        xticks=None,
        yticks=None,
        ticklabel_alternate_x=False,
        ticklabel_alternate_y=False,
        ticklabel_rotate_y=0,
        ticklabel_prune_x=None,
        ticklabel_prune_y=None,
        rotate_axes_180=False,
        style=None,
        rc_params={},
        filename=None,
        tight=True,
        plot3d=None,
        plot3d_filename=None,
        plot3d_texture=None,
        plot3d_alpha=None):

    if style is not None:
        matplotlib_styles.load_style(style)

    for k in rc_params.keys():
        matplotlib.rcParams[k] = rc_params[k]

    if ax is None:
        plot = MapPlot(figsize = figsize)
    else:
        plot = MapPlot(ax=ax, figsize = figsize)

    if background is not None:
        bg_img = plot.imshow(background, cmap=background_cmap, extent=extent, vmin=background_min_range, vmax=background_max_range, interpolation=interpolation)

    img = plot.imshow(X, primary=True, alpha=alpha, extent=extent, interpolation=interpolation)

    if cmap is not None:
        img.set_cmap(cmap)

    if colorbar is not None:
        assert colorbar == 'horizontal' or colorbar == 'vertical'
        if colorbar_location is None:
            if colorbar == 'horizontal':
                colorbar_location = 'bottom'
            else:
                colorbar_location = 'right'

        if colorbar == 'horizontal':
            label_location = 'top'
        else:
            label_location = 'left'

        plot.set_colorbar(
            discrete=discrete,
            rasterize=colorbar_rasterize,
            label=colorbar_label,
            label_size='medium',
            orientation=colorbar,
            location=colorbar_location,
            label_location=label_location,
            size=colorbar_size,
            pad=colorbar_pad,
            ticklabel_format=colorbar_ticklabel_format,
            ticklabel_size='small',
            min_range=min_range,
            max_range=max_range,
            num_classes=num_classes,
            class_boundaries=class_boundaries,
            extend=colorbar_extend,
            extendfrac=colorbar_extendfrac
        )

        if colorbar_shift_ticks:
            plot.shift_colorbar_ticks()

        if colorbar_hide_ticklines:
            for tick in plot.colorbar.ax.xaxis.get_major_ticks():
                tick.tick1On = False
                tick.tick2On = False

    if rotate_axes_180:
        plot.axes.invert_xaxis()
        plot.axes.invert_yaxis()

    plot.set_ticklabel_options(
        size='xx-small',
        xticks=xticks,
        yticks=yticks,
        alternate_x=ticklabel_alternate_x,
        alternate_y=ticklabel_alternate_y,
        prune_x=ticklabel_prune_x,
        prune_y=ticklabel_prune_y,
        rotate_y=ticklabel_rotate_y
    )

    # PDF output with semitransparent foreground images doesn't work with mpl 1.2.x
    # (only the foreground image is visible, ignoring the alpha value)
    # This bug might be responsible for this behavior: https://github.com/matplotlib/matplotlib/pull/1894
    # If pdf_transparency_bugfix == True, we add an additional RGB image on top of the existing
    # foreground image where the two images are already overlayed.
    if pdf_transparency_bugfix:
        bg_rgb = plot.get_image_rgb(bg_img)
        fg_rgb = plot.get_image_rgb(img)
        overlay_rgb = blend_images(bg_rgb, fg_rgb, alpha)
        plot.imshow(overlay_rgb, extent = extent)

    if filename is not None:
        savefig_kwargs = {}

        if tight:
            savefig_kwargs['bbox_inches'] = 'tight'

        if dpi is not None:
            savefig_kwargs['dpi'] = dpi

        plot.savefig(filename, **savefig_kwargs)

    if isinstance(plot3d, MapPlot3D):
        if plot3d_texture is not None:
            texture_width = plot3d_texture.shape[1]
            texture_height = plot3d_texture.shape[0]

            rgb = plot.get_image_rgb(plot.primary_image)
            if rgb.shape[0:2] != (texture_height, texture_width):
                rgb = resize_image(rgb, texture_width, texture_height)

            if plot3d_alpha is None:
                plot3d_alpha = np.ones(rgb.shape[0:2])

            plot3d.set_overlay(blend_images(plot3d_texture, rgb, plot3d_alpha))

        if plot3d_filename is not None:
            if plot3d_filename == '_auto_':
                plot3d_filename = '.'.join(filename.split('.')[:-1]) + '.png'

            plot3d.render(plot3d_filename)

    return plot

#def mapshow3d(plot, dtm,
#        mask = None,
#        texture = None,
#        z_exaggeration = 1.,
#        camera_settings = {},
#        filename = None,
#        crop = True):
#
#    plot3d = MapPlot3D(dtm = dtm)
#    rgb = plot.get_image_rgb(plot.primary_image)
#    width, height = rgb.shape[1], rgb.shape[0]
#
#    if texture is not None:
#        texture = Image.open(texture)
#        width, height = texture.size
#        rgb = resize_image(rgb, width, height)
#
#    if mask is not None:
#        mask, _ = read_gdal_file(mask)
#        mask = resize_image(mask, width, height)
#    else:
#        mask = np.ones((height, width))
#
#    overlay = np.asarray(texture).copy()
#    pos = (mask > 0)
#    for i in range(3): (overlay[:, :, i])[pos] = (rgb[:, :, i])[pos]
#
#    plot3d.set_overlay(overlay)
#    plot3d.z_exaggeration = z_exaggeration
#
#    plot3d.camera_settings = camera_settings
#
#    if filename is not None:
#        plot3d.render(filename)
#
#    return plot3d

def cmap_color_alphabet(N):
    """
    Creates a colormap using a set of N (N <= 26) colors for the Colour Alphabet Project
    suggested by Paul Green-Armytage in "A Colour Alphabet and the Limits of
    Colour Coding."

    See also:
    http://eleanormaclure.files.wordpress.com/2011/03/colour-coding.pdf
    http://graphicdesign.stackexchange.com/questions/3682/large-color-set-for-coloring-of-many-datasets-on-a-plot
    http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
    """

    assert N <= 26

    colors = np.array([[240, 163, 255],
            [0, 117, 220],
            [153, 63, 0],
            [76, 0, 92],
            [25, 25, 25],
            [0, 92, 49],
            [43, 206, 72],
            [255, 204, 153],
            [128, 128, 128],
            [148, 255, 181],
            [143, 124, 0],
            [157, 204, 0],
            [194, 0, 136],
            [0, 51, 128],
            [255, 164, 5],
            [255, 168, 187],
            [66, 102, 0],
            [255, 0, 16],
            [94, 241, 242],
            [0, 153, 143],
            [224, 255, 102],
            [116, 10, 255],
            [153, 0, 0],
            [255, 255, 128],
            [255, 255, 0],
            [255, 80, 5]]) / 255.

    return matplotlib.colors.ListedColormap(colors[0:N])

class MapPlot(object):
    def __init__(self, **kwargs):
        if 'ax' in kwargs:
            self.axes = kwargs['ax']
        else:
            fig = plt.figure(**kwargs)
            self.axes = fig.gca()

        self.colorbar = None

        self.primary_image = None

        # tick labels on all four sides: (to disable on one side use label1On = False etc.)
        self.axes.get_xaxis().set_tick_params(label2On = True)
        self.axes.get_yaxis().set_tick_params(label2On = True)

        # turn off offsets for the axes labels:
        self.axes.ticklabel_format(axis = 'both', style = 'plain', useOffset = False)

        plt.tight_layout()

    def imshow(self, X, primary=False, extent=None, update_extent=False, **kwargs):
        if isinstance(X, np.ndarray):
            image = self.axes.imshow(X, extent = extent, **kwargs)
        elif isinstance(X, str):
            X, extent = read_gdal_file(X)
            image = self.axes.imshow(X, extent = extent, **kwargs)

        if primary or len(self.axes.images) == 0:
            self.primary_image = image

        return image

    def set_colorbar(
        self,
        label=None,
        label_size=None,
        orientation='horizontal',
        location=None,
        label_location=None,
        size='5%',
        pad=1 / 2.54,
        discrete=False,
        rasterize=False,
        min_range=None,
        max_range=None,
        num_classes=None,
        class_boundaries=None,
        ticklabels=None,
        ticklabel_format=None,
        ticklabel_size=None,
        extend='neither',
        extendfrac=None
    ):
        image = self.primary_image
        cmap = image.cmap
        image_array = image.get_array()

        if min_range is None: min_range = image_array.min()
        if max_range is None: max_range = image_array.max()

        if location is None:
            if orientation == 'horizontal':
                location = 'bottom'
            elif orientation == 'vertical':
                location = 'right'

        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes(location, size = size, pad = pad)

        if discrete:
            if num_classes is None:
                num_classes = 10

            if class_boundaries is None:
                class_boundaries = min_range + np.array(range(num_classes + 1)) * (max_range - min_range) / float(num_classes)

            num_classes = len(class_boundaries) - 1

            if cmap.N != num_classes:
                cmap_pos = np.array(range(num_classes)) / float(num_classes - 1)
                cmap = matplotlib.colors.ListedColormap(cmap(cmap_pos))
            image.cmap = cmap

            norm = matplotlib.colors.BoundaryNorm(class_boundaries, cmap.N)
            image.set_norm(norm)

            self.colorbar=matplotlib.colorbar.ColorbarBase(
                cax,
                cmap=cmap,
                norm=norm,
                orientation=orientation,
                boundaries=class_boundaries,
                spacing='uniform',
                extend=extend,
                extendfrac=extendfrac,
                ticks=class_boundaries
            )
        else:
            self.colorbar = plt.colorbar(
                image,
                cmap=cmap,
                cax=cax,
                orientation=orientation
            )
            image.set_clim(min_range, max_range)

        cax.tick_params(direction='out')
        ax = cax.xaxis if orientation == 'horizontal' else cax.yaxis
        self._primary_colorbar_axis = ax

        if label is not None:
            kwargs = {} if label_size is None else {'size': label_size}
            self.colorbar.set_label(label, **kwargs)
            if label_location is not None: ax.set_label_position(label_location)

        if ticklabels is None and ticklabel_format is not None and not discrete:
            # "and not discrete" because of the uniform spacing for the discrete colorbars, i.e. the values do not (necessarily) correspond to the tick locations
            tick_locations = ax.get_ticklocs() # get tick locations in data coordinates
            vmin = self.colorbar.vmin
            vmax = self.colorbar.vmax
            tick_values = tick_locations * (vmax - vmin) + vmin # tick locations in display coordinates
            ticklabels = [ticklabel_format % t for t in tick_values]
            ax.set_ticklabels(ticklabels)

        if ticklabel_size is not None:
            for t in ax.get_ticklabels(): t.set_fontsize(ticklabel_size)

        self.colorbar.solids.set_rasterized(rasterize) # set rasterize to True to remove artifacts in PDFs

        # display solid colorbar (and not transparent with the same alpha value as the image itself):
        # (this also removes the artifacts (lines) that sometimes occur)
        # (from http://stackoverflow.com/questions/15003353/why-does-my-colorbar-have-lines-in-it)
        self.colorbar.set_alpha(1)
        self.colorbar.draw_all()
        self.colorbar.solids.set_edgecolor('face')

    def set_alternating_xaxis_ticklabels(self):
        ax = self.axes.get_xaxis()
        ticks = ax.get_major_ticks()
        var = False
        for tick in ticks:
            tick.label1On = var
            tick.label2On = not var
            var = not var

    def set_ticklabel_options(
        self,
        xticks=None,
        yticks=None,
        size=None,
        left=True,
        right=True,
        bottom=True,
        top=True,
        alternate_x=False,
        alternate_y=False,
        prune_x=None,
        prune_y=None,
        rotate_y=0
    ):
        xaxis = self.axes.get_xaxis()
        yaxis = self.axes.get_yaxis()

        if xticks is not None:
            self.axes.set_xticks(xticks)

        if yticks is not None:
            self.axes.set_yticks(yticks)

        if size is not None:
            for tick in xaxis.get_ticklabels() + yaxis.get_ticklabels():
                tick.set_fontsize(size)

        xaxis.set_tick_params(label1On=bottom, label2On=top)
        yaxis.set_tick_params(label1On=left, label2On=right)

        ticks = []
        var = False
        if alternate_x: ticks += xaxis.get_major_ticks()
        if alternate_y: ticks += yaxis.get_major_ticks()
        for tick in ticks:
            tick.label1On = var
            tick.label2On = not var
            var = not var

        # remove overlapping ticklabels (http://stackoverflow.com/questions/9422587/overlapping-y-axis-tick-label-and-x-axis-tick-label-in-matplotlib):
        if prune_x is not None:
            xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(prune=prune_x))
        if prune_y is not None:
            yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(prune=prune_y))

        if rotate_y != 0:
            for tick in yaxis.get_ticklabels(): tick.set_rotation(rotate_y)

    def savefig(self, filename, **kwargs):
        self.axes.get_figure().savefig(filename, **kwargs)

    def get_image_rgb(self, image):
        shape = image.get_array().shape

        fig = plt.figure(figsize=(shape[1], shape[0]), dpi=1)
        axes = fig.gca()
        axes.axis('off')
        axes.set_position([0, 0, 1, 1]) # remove bounding box

        image_new = axes.imshow(image.get_array())
        image_new.set_cmap(image.get_cmap())
        image_new.set_norm(image.norm)

        canvas = fig.canvas
        canvas.draw()
        data = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return data

    def shift_colorbar_ticks(self):
        """
        Shift the ticks of the colorbar by one half of the distance between the
        tick. Can be useful for discrete colorbars to label the actual colors and
        not the boundaries between the colors.
        Example:
        plot = mapplot.mapshow(
                (...),
                discrete = True,
                min_range = 1,
                max_range = 19,
                num_classes = 18)
        Here without using shift_colorbar_ticks, the ticks would be 0...19 from
        bottom to top of the colorbar. With shift_colorbar_ticks, the highest
        tick (19) is removed, and the other ticks are shifted upwards.
        """
        ax = self._primary_colorbar_axis
        ticklocs = ax.get_ticklocs()
        ticklabels = [tl.get_text() for tl in ax.get_ticklabels()[:-1]]
        shift = (ticklocs[1] - ticklocs[0]) / 2.
        ticklocs = [t + shift for t in ticklocs[:-1]]
        ax.set_ticks(ticklocs)
        for i, tl in enumerate(ax.get_ticklabels()):
            tl.set_text(ticklabels[i])





class MapPlot3D(object):
    def __init__(
        self,
        dtm=None,
        overlay=None,
        sample_rate=10,
        z_exaggeration=1.,
        background=(1, 1, 1)
    ):
        if dtm is not None: self.set_dtm(dtm)
        if overlay is not None: self.set_overlay(overlay)
        self.sample_rate = sample_rate
        self.z_exaggeration = z_exaggeration
        self.camera_settings = {}

    def set_dtm(self, dtm):
        assert type(dtm) in (str, unicode)
        self._dtm_reader = tvtk.DataSetReader(file_name = dtm)
        self._dtm_reader.update()
        (self._x_min, self._x_max,
            self._y_min, self._y_max,
            self._z_min, self._z_max) = self._dtm_reader.output.extent #whole_extent

    def set_overlay(self, overlay):
        assert isinstance(overlay, np.ndarray)
        dims = self._dtm_reader.output.dimensions
        self._overlay = vtkutils.vtk_image_from_array(overlay)

    def render(self, filename=None, crop=True, add_actors=[]):
        renderer = tvtk.Renderer(background = (1, 1, 1))

        subset = tvtk.ExtractVOI(
                sample_rate = (self.sample_rate, self.sample_rate, 1),
                voi = (self._x_min, self._x_max, self._y_min, self._y_max, 0, 0))
        configure_input(subset, self._dtm_reader)

        cropped = tvtk.Threshold()
        configure_input(cropped, subset)
        cropped.threshold_by_upper(-9998)

        geom = tvtk.GeometryFilter()
        configure_input(geom, cropped)
        surface1 = tvtk.WarpScalar(scale_factor = self.z_exaggeration)
        configure_input(surface1, geom)
        triangles = tvtk.TriangleFilter()
        configure_input(triangles, surface1)
        reduced = tvtk.DecimatePro(target_reduction = 0.5, preserve_topology = True)
        configure_input(reduced, triangles)
        strips = tvtk.Stripper()
        configure_input(strips, reduced)
        texture_plane = tvtk.TextureMapToPlane(
                origin = (0, self._y_max, 0),
                point1 = (self._x_max, self._y_max, 0),
                point2 = (0, 0, 0))
        configure_input(texture_plane, strips)

        map_normals = tvtk.PolyDataNormals()
        configure_input(map_normals, texture_plane)
        surface = tvtk.PolyDataMapper(scalar_visibility = False)
        configure_input(surface, map_normals)
        #map_image = tvtk.PNGReader(file_name = overlay_file)
        map_texture = tvtk.Texture(interpolate = True)
        configure_input(map_texture, self._overlay)

        geomap_actor = tvtk.Actor(mapper = surface, texture = map_texture, visibility = True)
        renderer.add_actor(geomap_actor)

        overhead_light = tvtk.Light()
        overhead_light.set_color(1, 1, 1)
        overhead_light.focal_point = ((self._x_max - self._x_min) / 2, (self._y_max - self._y_min) / 2, 0)
        overhead_light.position = ((self._x_max - self._x_min) / 2, (self._y_max - self._y_min) / 2, 2000)
        overhead_light.intensity = 0.5
        renderer.add_light(overhead_light)

        headlight = tvtk.Light(light_type = 'headlight')
        renderer.add_light(headlight)

        renderer.active_camera.set(**self.camera_settings)

        ## add axes indicator:
        #axes = tvtk.AxesActor()
        #axes.total_length = (300,) * 3
        #text_property = tvtk.TextProperty()
        #text_property.font_family = 'arial'
        #text_property.font_size = 40
        #text_property.color = (0, 0, 0)
        #axes.x_axis_caption_actor2d.text_actor.text_scale_mode = False
        #axes.y_axis_caption_actor2d.text_actor.text_scale_mode = False
        #axes.z_axis_caption_actor2d.text_actor.text_scale_mode = False
        #axes.x_axis_caption_actor2d.caption_text_property = text_property
        #axes.y_axis_caption_actor2d.caption_text_property = text_property
        #axes.z_axis_caption_actor2d.caption_text_property = text_property
        #axes.x_axis_caption_actor2d.position = [10, 0]
        #axes.y_axis_caption_actor2d.position = [20, -20]
        #axes.z_axis_caption_actor2d.position = [20, 0]
        #axes.x_axis_shaft_property.line_width = 3
        #axes.y_axis_shaft_property.line_width = 3
        #axes.z_axis_shaft_property.line_width = 3
        #renderer.add_actor(axes)

        ## shift it a little bit:
        #transform = tvtk.Transform()
        #transform.translate((400, 400, 0))
        #axes.user_transform = transform

        # add additional actors:
        for actor in add_actors:
            renderer.add_actor(actor)

        if filename is None:
            v = ivtk.viewer()
            v.scene.add_actors(renderer.actors)
            cam = v.scene.camera
        else:
            render_window = tvtk.RenderWindow(size = (840, 700), position = (100, 100))
            render_window.add_renderer(renderer)
            cam = renderer.active_camera

        if filename is not None:
            render_window.render()
            render_large = tvtk.RenderLargeImage(input = renderer, magnification = 2)
            writer = tvtk.PNGWriter(file_name = filename)
            configure_input(writer, render_large)
            writer.write()

            if crop:
                # crop image to non-white pixels:
                image = Image.open(filename)
                bg = Image.new(image.mode, image.size, (255, 255, 255))
                diff = ImageChops.difference(image, bg)
                bbox = diff.getbbox()
                image = image.crop(bbox)
                image.save(filename)
