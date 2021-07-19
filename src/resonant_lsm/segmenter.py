import fnmatch
import re
import yaml
import sys
import os
from collections import MutableMapping

import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np


class FixedDict(MutableMapping):
    def __init__(self, data):
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __iter__(self):
        return iter(self.__data)

    def __setitem__(self, k, v):
        if k not in self.__data:
            raise KeyError("{:s} is not an acceptable key.".format(k))

        self.__data[k] = v

    def __delitem__(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        if k not in self.__data:
            raise KeyError("{:s} is not an acceptable key.".format(k))
        return self.__data[k]

    def __contains__(self, k):
        return k in self.__data

    def __repr__(self):
        return repr(self.__data)


class segmenter():
    def __init__(self, **kwargs):
        """
        Segments cells indicated by user-supplied seed points.

        Parameters
        ----------
        config : str
            The path to a configuration file where all other keyword arguments can be specified.
        image_directory : str
            Path to image files.
        spacing : [int, int, int]
            Image spacing.
        bounding_box : [int, int, int]
            Region to consider around each seed point during semgmentation.
        seed_points : [[int, int, int],...[int,int, int]]
            List of points near approximate centroids of cells to segment.

        Attributes
        ----------
        image : SimpleITK.Image()
            3-D image for object segmentation.
        levelsets : [SimpleITK.Image(), ...]
            List of levelset images returned by active contour models.
        isocontours : [vtk.PolyData(), ...]
            List of polygonal surfaces of segmented objects
        """

        self.options = FixedDict({
            'image_directory': None,
            'spacing': [1.0, 1.0, 1.0],
            'bounding_box': [100, 100, 20],
            'seed_points': None, })
        self.config = None
        for key, value in kwargs.items():
            if key in self.options.keys():
                setattr(self.options, key, value)
            else:
                setattr(self, key, value)

        if self.config is not None:
            self.parseConfig()

        self.output_dir = '_'.join([self.options['image_directory'], 'results'])
        self.image = None
        self.levelsets = []
        self.isocontours = []

    def execute(self):
        self._castOptions()
        self.parseStack()
        output_dir = '_'.join([self.options['image_directory'], 'results'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.writeImageAsVTK(os.path.join(output_dir, 'image'))
        for i, s in enumerate(self.options['seed_points']):
            origin = [int(x - b / 2.0 - 1) for (x, b) in zip(s, self.options['bounding_box'])]
            tmp_bounding_box = np.copy(self.options['bounding_box'])
            for j in range(3):
                if origin[j] < 0:
                    origin[j] = 0
                elif origin[j] > self.image.GetSize()[j] - 1:
                    origin[j] = self.image.GetSize()[j] - 1
                if origin[j] + tmp_bounding_box[j] > self.image.GetSize()[j] - 1:
                    tmp_bounding_box[j] -= origin[j] + tmp_bounding_box[j] - self.image.GetSize()[j]
            roi = sitk.RegionOfInterest(self.image, tmp_bounding_box.tolist(), origin)
            self.segmentROI(roi, [s], i)
            self.makeSurface(self.levelsets[-1])
            self.writeSurface(self.isocontours[-1], i + 1)

    def _castOptions(self):
        arrays = (('spacing', 'float'),
                  ('bounding_box', 'int'),
                  ('seed_points', 'int'))
        for k, v in arrays:
            if v == 'float':
                self.options[k] = np.array(self.options[k], float)
            elif v == 'int':
                self.options[k] = np.array(self.options[k], int)

    def parseConfig(self):
        with open(self.config) as user:
            user_settings = yaml.load(user, yaml.SafeLoader)

        for k, v in list(user_settings.items()):
            self.options[k] = v

    def parseStack(self):
        directory = self.options['image_directory']
        for ftype in ['*.nii', '*.tif*']:
            files = fnmatch.filter(sorted(os.listdir(directory)), ftype)
            if len(files) > 0:
                break

        if ftype == "*.tif*":
            counter = [re.search("[0-9]*\.tif", f).group() for f in files]
            for i, c in enumerate(counter):
                counter[i] = int(c.replace('.tif', ''))
            files = np.array(files, dtype=object)
            sorter = np.argsort(counter)
            files = files[sorter]
            img = []
            for fname in files:
                filename = os.path.join(directory, fname)
                img.append(sitk.ReadImage(filename, sitk.sitkFloat32))
            img = sitk.RescaleIntensity(sitk.JoinSeries(img), 0.0, 1.0)
            print(("\nImported 3D image stack ranging from {:s} to {:s}".format(
                files[0], files[-1])))
        elif ftype == "*.nii":
            filename = os.path.normpath(directory, files[0])
            img = sitk.ReadImage(filename, sitk.sitkFloat32)
            img = sitk.RescaleIntensity(img, 0.0, 1.0)

        img.SetSpacing(self.options['spacing'])
        self.image = img

    def segmentROI(self, roi, seedpoint, counter):

        smoothed = sitk.RescaleIntensity(roi, 0, 1)
        print('... Cell {:03d}'.format(counter + 1))
        print('... ... Adjusting Contrast')
        smoothed = sitk.AdaptiveHistogramEqualization(smoothed,
                                                      radius=[int(s / f) for s, f in
                                                              zip(smoothed.GetSize(),
                                                                  (4, 4, 4))])
        print('... ... Resampling to isotropic voxel')
        res = sitk.ResampleImageFilter()
        spacing = np.min(smoothed.GetSpacing())
        ratio = [orig / spacing for orig in smoothed.GetSpacing()]
        size = [int(np.ceil(r * sz)) for r, sz in
                zip(ratio, smoothed.GetSize())]
        res.SetOutputOrigin(smoothed.GetOrigin())
        res.SetSize(size)
        res.SetOutputSpacing([spacing] * 3)
        res.SetInterpolator(sitk.sitkLinear)
        smoothed = res.Execute(smoothed)
        seedpoint = [[int(np.ceil((x - o / s) * r)) for x, o, s, r in zip(seedpoint[0],
                      roi.GetOrigin(), roi.GetSpacing(), ratio)]]
        initial = sitk.Cast(smoothed * 0, sitk.sitkUInt8) + 1
        for i in range(smoothed.GetSize()[0]):
            for j in range(smoothed.GetSize()[1]):
                initial[i, j, 0] = 0
                initial[i, j, smoothed.GetSize()[2] - 1] = 0
        for i in range(smoothed.GetSize()[0]):
            for j in range(smoothed.GetSize()[2]):
                initial[i, 0, j] = 0
                initial[i, smoothed.GetSize()[1] - 1, j] = 0
        for i in range(smoothed.GetSize()[1]):
            for j in range(smoothed.GetSize()[2]):
                initial[0, i, j] = 0
        print('... ... Segmenting')
        ls = sitk.ScalarChanAndVeseDenseLevelSet(sitk.Cast(initial, sitk.sitkFloat32),
                                                 smoothed * 255,
                                                 maximumRMSError=0.02,
                                                 epsilon=1.5,
                                                 numberOfIterations=100,
                                                 lambda1=0.5,
                                                 curvatureWeight=0.0,
                                                 reinitializationSmoothingWeight=0.0)
        binary = sitk.BinaryThreshold(ls, 0.1, 1e7)
        binary = sitk.BinaryMorphologicalOpening(binary, [1, 1, 1])
        binary = sitk.BinaryFillhole(binary)
        components = sitk.ConnectedComponent(binary)
        labelstats = sitk.LabelShapeStatisticsImageFilter()
        labelstats.Execute(components)
        labels = labelstats.GetLabels()
        volumes = []
        for label in labels:
            volumes.append(labelstats.GetPhysicalSize(label))
            if components[seedpoint[0]] == label:
                best_label = label
        try:
            best_label
        except NameError:
            print("No Labels in image")
            best_label = np.argmax(volumes) + 1
        mask = sitk.Image(*components.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(binary)
        for label in labels:
            if label != best_label:
                mask = mask + sitk.Cast(components == label,
                                        sitk.sitkUInt8)
        mask2 = sitk.BinaryDilate(components == best_label, [4, 4, 4])
        mask2 = sitk.InvertIntensity(mask2, 1)
        mask = sitk.BinaryThreshold(ls, 0.1, 1e7)
        mask = mask * mask2
        mask = sitk.InvertIntensity(mask, 1)
        ls = ls * sitk.Cast(mask, sitk.sitkFloat32)
        ls = sitk.SmoothingRecursiveGaussian(ls, 1.0)
        self.levelsets.append(ls)

    def writeImageAsVTK(self, name):
        """
        Save image to disk as a .vti file. Image will be resampled such that it has spacing equal
        to that specified in *options["Image"]["spacing"]*.

        Parameters
        ----------
        name : str, required
            Name of file to save to disk without the file suffix
        """
        print("... Saving Image to {:s}.vti".format(name))
        a = numpy_to_vtk(sitk.GetArrayFromImage(self.image).ravel(),
                         deep=True, array_type=vtk.VTK_FLOAT)
        vtk_img = vtk.vtkImageData()
        vtk_img.SetOrigin(self.image.GetOrigin())
        vtk_img.SetSpacing(self.image.GetSpacing())
        vtk_img.SetDimensions(self.image.GetSize())
        vtk_img.GetPointData().SetScalars(a)
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName("{:s}.vti".format(name))
        writer.SetInputData(vtk_img)
        writer.Write()

    def makeSurface(self, ls):
        origin = list(ls.GetOrigin())
        spacing = list(ls.GetSpacing())
        dimensions = list(ls.GetSize())

        vtkimage = vtk.vtkImageData()
        vtkimage.SetOrigin(origin)
        vtkimage.SetSpacing(spacing)
        vtkimage.SetDimensions(dimensions)

        pixel_type = {1: vtk.VTK_UNSIGNED_CHAR,
                      3: vtk.VTK_UNSIGNED_INT,
                      8: vtk.VTK_FLOAT,
                      9: vtk.VTK_DOUBLE}

        intensities = numpy_to_vtk(sitk.GetArrayFromImage(ls).ravel(), deep=True,
                                   array_type=pixel_type[ls.GetPixelID()])
        intensities.SetName("Intensity")
        intensities.SetNumberOfComponents(1)

        vtkimage.GetPointData().SetScalars(intensities)
        vtkimage = vtkimage
        iso = vtk.vtkContourFilter()
        iso.SetInputData(vtkimage)
        iso.ComputeScalarsOff()
        iso.ComputeNormalsOn()
        iso.SetValue(0, 0.3)
        iso.Update()

        triangles = vtk.vtkGeometryFilter()
        triangles.SetInputConnection(iso.GetOutputPort())
        triangles.Update()

        # check for holes
        boundaries = vtk.vtkFeatureEdges()
        boundaries.ManifoldEdgesOff()
        boundaries.FeatureEdgesOff()
        boundaries.NonManifoldEdgesOff()
        boundaries.BoundaryEdgesOn()
        boundaries.SetInputConnection(triangles.GetOutputPort())
        boundaries.Update()
        edges = vtk.vtkGeometryFilter()
        edges.SetInputConnection(boundaries.GetOutputPort())
        edges.Update()
        if edges.GetOutput().GetNumberOfLines() > 0:
            print('::WARNING:: Hole(s) detected in isocontour. These were filled with a planar cap.')
        deci = vtk.vtkDecimatePro()
        deci.SetInputConnection(triangles.GetOutputPort())
        deci.PreserveTopologyOff()
        deci.SplittingOff()
        deci.BoundaryVertexDeletionOn()
        deci.SetTargetReduction(0.1)
        fillhole = vtk.vtkFillHolesFilter()
        fillhole.SetHoleSize(1e7)
        fillhole.SetInputConnection(deci.GetOutputPort())
        fillhole.Update()
        smooth = vtk.vtkWindowedSincPolyDataFilter()
        smooth.SetInputConnection(fillhole.GetOutputPort())
        smooth.NormalizeCoordinatesOn()
        smooth.SetNumberOfIterations(30)
        smooth.SetPassBand(0.01)
        smooth.FeatureEdgeSmoothingOff()
        smooth.Update()

        normal_generator = vtk.vtkPolyDataNormals()
        normal_generator.ComputePointNormalsOn()
        normal_generator.SetInputConnection(smooth.GetOutputPort())
        normal_generator.Update()
        isocontour = normal_generator.GetOutput()

        self.isocontours.append(isocontour)

    def writeSurface(self, iso, counter):
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(os.path.join(self.output_dir, 'cell_{:03d}.stl'.format(counter)))
        writer.SetInputData(iso)
        writer.Write()


if __name__ == '__main__':
    seg = segmenter(config=sys.argv[-1])
    seg.execute()
