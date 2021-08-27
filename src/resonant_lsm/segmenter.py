import fnmatch
import re
import yaml
import sys
import os

import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np


class segmenter:
    def __init__(self,
                 config: str = '',
                 image_directory: str = '',
                 spacing: list[float, float, float] = (1.0, 1.0, 1.0),
                 equalization_fraction: float = 0.25,
                 levelset_smoothing_radius: int = 2,
                 curvature_weight: float = 10.0,
                 area_weight: float = 50.0,
                 bounding_box: list[int] = (100, 100, 20),
                 seed_points: list[list[int, int, int]] = ()):
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
        equalization_fraction : float
            Window size use for histogram equalization as a fraction of regions of interest edge lengths.
            0 to turn off equalization.
        levelset_smoothing_radius : int
            Radius in voxels of median filter used for smoothing levelsets. No smoothing performed if 0.
        curvature_weight : float
            Penalizes curvature of the levelset solution in active contour model. Higher results in smoother
            segmentation, but more likely to overestimate volume or fail to separate close objects.
        area_weight : float
            Penalizes area (volume in 3d) of levelset solution. Higher tends to result in a smaller segmented volume.
        bounding_box : [int, int, int]
            Region to consider around each seed point during segmentation.
        seed_points : [[int, int, int],...[int,int, int]]
            List of points near approximate centroids of cells to segment.

        Attributes
        ----------
        output_dir : str
            Directory where results are written to disk.
        image : SimpleITK.Image()
            3-D image for object segmentation.
        levelsets : [SimpleITK.Image(), ...]
            List of levelset images returned by active contour models.
        isocontours : [vtk.PolyData(), ...]
            List of polygonal surfaces of segmented objects
        """

        self.config = config
        self.image_directory = image_directory
        self.spacing = spacing
        self.equalization_fraction = equalization_fraction
        self.levelset_smoothing_radius = levelset_smoothing_radius
        self.curvature_weight = curvature_weight
        self.area_weight = area_weight
        self.bounding_box = bounding_box
        self.seed_points = seed_points

        if len(self.config) > 0:
            self.parse_config()

        self.image = None
        self.output_dir = '_'.join([self.image_directory, 'results'])
        self.levelsets = []
        self.isocontours = []

    def execute(self):
        self.image = self.parse_stack()
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.write_image_as_vtk(self.image, os.path.join(self.output_dir, 'image'))
        for i, s in enumerate(self.seed_points):
            origin = [int(x - b / 2.0 - 1) for (x, b) in zip(s, self.bounding_box)]
            tmp_bounding_box = np.copy(self.bounding_box)
            for j in range(3):
                if origin[j] < 0:
                    origin[j] = 0
                elif origin[j] > self.image.GetSize()[j] - 1:
                    origin[j] = self.image.GetSize()[j] - 1
                if origin[j] + tmp_bounding_box[j] > self.image.GetSize()[j] - 1:
                    tmp_bounding_box[j] -= origin[j] + tmp_bounding_box[j] - self.image.GetSize()[j]
            roi = sitk.RegionOfInterest(self.image, tmp_bounding_box.tolist(), origin)
            self.segment_region_of_interest(roi, [s], i)
            self.make_surface(self.levelsets[-1])
            self.write_surface(self.isocontours[-1], i + 1)

    def parse_config(self):
        with open(self.config) as user:
            user_settings = yaml.load(user, yaml.SafeLoader)

        for k, v in list(user_settings.items()):
            setattr(self, k, v)

    def parse_stack(self):
        directory = self.image_directory
        for file_type in ['*.nii', '*.tif*']:
            files = fnmatch.filter(sorted(os.listdir(directory)), file_type)
            if len(files) > 0:
                break

        if file_type == "*.tif*":
            counter = [re.search("[0-9]*\.tif", f).group() for f in files]
            for i, c in enumerate(counter):
                counter[i] = int(c.replace('.tif', ''))
            files = np.array(files, dtype=object)
            sorter = np.argsort(counter)
            files = files[sorter]
            img = []
            for f_name in files:
                filename = os.path.join(directory, f_name)
                img.append(sitk.ReadImage(filename, sitk.sitkFloat32))
            img = sitk.RescaleIntensity(sitk.JoinSeries(img), 0.0, 1.0)
            print(("\nImported 3D image stack ranging from {:s} to {:s}".format(
                files[0], files[-1])))
        elif file_type == "*.nii":
            filename = os.path.join(directory, files[0])
            img = sitk.ReadImage(filename, sitk.sitkFloat32)
            img = sitk.RescaleIntensity(img, 0.0, 1.0)
        else:
            raise TypeError("The directory must contain either a sequence of TIFF images or a single"
                            " NifTi (.nii) image.")

        img.SetSpacing(self.spacing)
        return img

    def segment_region_of_interest(self, roi, seedpoint, counter):
        roi = sitk.RescaleIntensity(roi, 0, 1)
        print('... Cell {:03d}'.format(counter + 1))
        if self.equalization_fraction > 0:
            equalized = self._apply_adaptive_histogram_equalization(roi)
        else:
            equalized = roi
        seedpoint, resampled = self._resample(seedpoint, equalized)
        initial = self._create_initial_levelset(resampled)
        print('... ... Segmenting')
        sc = sitk.ScalarChanAndVeseDenseLevelSetImageFilter()
        sc.SetLambda1(1.0)
        sc.SetEpsilon(2.0)
        sc.SetCurvatureWeight(self.curvature_weight)
        sc.SetAreaWeight(self.area_weight)
        sc.UseImageSpacingOff()
        sc.SetNumberOfIterations(100)
        sc.SetMaximumRMSError(0.02)
        ls = sc.Execute(sitk.Cast(initial, sitk.sitkFloat32),
                        resampled * 255)

        print('... ... ... Final Metric Value: {}'.format(sc.GetRMSChange()))
        print('... ... ... Number of iterations: {}'.format(sc.GetNumberOfIterations()))
        cell_mask = self._isolate_cell(ls, seedpoint)

        ls = ls * sitk.Cast(cell_mask, sitk.sitkFloat32)
        if self.levelset_smoothing_radius > 0:
            ls = sitk.Median(ls, [radius] * 3)
        self.levelsets.append(ls)

    def _apply_adaptive_histogram_equalization(self, image):
        print('... ... Adjusting Contrast')
        fraction = self.equalization_fraction
        return sitk.AdaptiveHistogramEqualization(image,
                                                  radius=[int(s * fraction) for s in image.GetSize()],
                                                  alpha=0.6,
                                                  beta=0.2)

    def _isolate_cell(self, levelset, seedpoint):
        best_label = -1
        binary = sitk.BinaryThreshold(levelset, 0.7, 1e7)
        binary = sitk.BinaryMorphologicalOpening(binary, [4, 4, 4])
        binary = sitk.BinaryFillhole(binary)
        components = sitk.ConnectedComponent(binary)
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(components)
        labels = label_stats.GetLabels()
        volumes = []
        for label in labels:
            volumes.append(label_stats.GetPhysicalSize(label))
            if components[seedpoint[0]] == label:
                best_label = label
        if best_label < 0:
            print("No Labels in image")
            best_label = np.argmax(volumes) + 1

        cell_mask = self._mask_cell(levelset, components, labels, best_label)
        return cell_mask

    @staticmethod
    def _mask_cell(levelset, components, labels, best_label):
        mask = sitk.Image(*components.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(components)
        for label in labels:
            if label != best_label:
                mask = mask + sitk.Cast(components == label,
                                        sitk.sitkUInt8)
        mask2 = sitk.BinaryDilate(components == best_label, [4, 4, 4])
        mask2 = sitk.InvertIntensity(mask2, 1)
        mask = sitk.BinaryThreshold(levelset, 0.1, 1e7)
        mask = mask * mask2
        mask = sitk.InvertIntensity(mask, 1)
        return mask

    @staticmethod
    def _create_initial_levelset(image):
        initial = sitk.Image(image.GetSize(), sitk.sitkUInt8) * 0
        initial.CopyInformation(image)
        for i in range(initial.GetSize()[0]):
            for j in range(initial.GetSize()[1]):
                initial[i, j, 0] = 1
                initial[i, j, initial.GetSize()[2] - 1] = 1
        for i in range(initial.GetSize()[0]):
            for j in range(initial.GetSize()[2]):
                initial[i, 0, j] = 1
                initial[i, initial.GetSize()[1] - 1, j] = 1
        for i in range(initial.GetSize()[1]):
            for j in range(initial.GetSize()[2]):
                initial[0, i, j] = 1
                initial[initial.GetSize()[0] - 1, i, j] = 1
        return initial

    def _resample(self, seedpoint, image):
        print('... ... Resampling to isotropic voxel')
        res = sitk.ResampleImageFilter()
        spacing = np.min(image.GetSpacing())
        ratio = [orig / spacing for orig in image.GetSpacing()]
        size = [int(np.ceil(r * sz)) for r, sz in
                zip(ratio, image.GetSize())]
        res.SetOutputOrigin(image.GetOrigin())
        res.SetSize(size)
        res.SetOutputSpacing([spacing] * 3)
        res.SetInterpolator(sitk.sitkLinear)
        resampled = res.Execute(image)
        index_origin = np.array(image.GetOrigin()) - np.array(self.image.GetOrigin())
        index_origin /= np.array(image.GetSpacing())
        seedpoint = [[int(np.ceil((x - o) * r)) for x, o, r in zip(seedpoint[0], index_origin, ratio)]]
        return seedpoint, resampled

    def write_image_as_vtk(self, image, name):
        """
        Save image to disk as a .vti file. Image will be resampled such that it has spacing equal
        to that specified in *options["Image"]["spacing"]*.

        Parameters
        ----------
        image: SimpleITK.Image
        name : str, required
            Name of file to save to disk without the file suffix
        """
        print("... Saving Image to {:s}.vti".format(name))
        vtk_img = self._convert_sitk_to_vtk(image)
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName("{:s}.vti".format(name))
        writer.SetInputData(vtk_img)
        writer.Write()

    @staticmethod
    def _convert_sitk_to_vtk(image):
        image = sitk.Cast(image, sitk.sitkFloat32)
        pixel_type = {1: vtk.VTK_UNSIGNED_CHAR,
                      3: vtk.VTK_UNSIGNED_INT,
                      8: vtk.VTK_FLOAT,
                      9: vtk.VTK_DOUBLE}
        intensities = numpy_to_vtk(sitk.GetArrayFromImage(image).ravel(),
                                   deep=True, array_type=pixel_type[image.GetPixelID()])
        intensities.SetName("Intensity")
        intensities.SetNumberOfComponents(1)
        vtk_img = vtk.vtkImageData()
        vtk_img.SetOrigin(image.GetOrigin())
        vtk_img.SetSpacing(image.GetSpacing())
        vtk_img.SetDimensions(image.GetSize())
        vtk_img.GetPointData().SetScalars(intensities)
        return vtk_img

    def make_surface(self, ls):
        binary = sitk.BinaryThreshold(ls, 0.1, 1e7)
        smoothed = sitk.AntiAliasBinary(binary)

        vtk_image = self._convert_sitk_to_vtk(smoothed)

        iso = vtk.vtkFlyingEdges3D()
        iso.SetInputData(vtk_image)
        iso.ComputeScalarsOff()
        iso.ComputeNormalsOn()
        iso.SetValue(0, -1.0)
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

    def write_surface(self, iso, counter):
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(os.path.join(self.output_dir, 'cell_{:03d}.stl'.format(counter)))
        writer.SetInputData(iso)
        writer.Write()


if __name__ == '__main__':
    seg = segmenter(config=sys.argv[-1])
    seg.execute()
