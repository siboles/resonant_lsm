import datetime
import os

import numpy as np
import pandas
import vtk
from resonant_lsm import segmenter, generate_images
from vtk.util.numpy_support import vtk_to_numpy

REPEATS = 5
GAUSSIAN_NOISE = [0.1, 0.2]
SPECKLE_NOISE = [0.0, 0.1, 0.2]


class DataComparison:
    def __init__(self, true_surface_path, segmented_surface):
        self.true_surface = self.read_vtk_polydata(true_surface_path)
        self.segmented_surface = segmented_surface
        self.true_volume, self.true_surface_area = self.get_volume_and_surface_area(self.true_surface)
        self.segmented_volume, self.segmented_surface_area = self.get_volume_and_surface_area(self.segmented_surface)

        self.error_polydata = self.get_shape_offset_error()
        self.shape_offset_rms_error = self.get_rms_error(self.error_polydata.GetPointData().GetScalars())

    @staticmethod
    def read_vtk_polydata(filepath):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        return reader.GetOutput()

    @staticmethod
    def get_volume_and_surface_area(surface):
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(surface)
        mass_properties.Update()
        return mass_properties.GetVolume(), mass_properties.GetSurfaceArea()

    def get_shape_offset_error(self):
        distance = vtk.vtkDistancePolyDataFilter()
        distance.SetInputData(0, self.true_surface)
        distance.SetInputData(1, self.segmented_surface)
        distance.Update()
        return distance.GetOutput()

    @staticmethod
    def get_rms_error(array):
        numpy_array = vtk_to_numpy(array)
        return np.linalg.norm(numpy_array) / np.sqrt(numpy_array.size)

    def write_polydata(self, name):
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(''.join([name, '.vtk']))
        writer.SetInputData(self.error_polydata)
        writer.Update()
        writer.Write()


now = datetime.datetime.now()
root_directory = now.strftime('tests_%m_%d_%H_%M')
os.mkdir(root_directory)


def update_results(results: dict, compare: DataComparison) -> dict:
    results['True Volume'].append(compare.true_volume)
    results['True Surface Area'].append(compare.true_surface_area)
    results['Segmented Volume'].append(compare.segmented_volume)
    results['Segmented Surface Area'].append(compare.segmented_surface_area)
    results['Volume Absolute Error'].append(compare.segmented_volume - compare.true_volume)
    results['Surface Area Absolute Error'].append(compare.segmented_surface_area - compare.true_surface_area)
    results['Volume Percentage Error'].append((compare.segmented_volume / compare.true_volume - 1.0) * 100.0)
    results['Surface Area Percentage Error'].append(
        (compare.segmented_surface_area / compare.true_surface_area - 1.0) * 100.0)
    results['Root Mean Square Offset Error'].append(compare.shape_offset_rms_error)
    return results


for gn in GAUSSIAN_NOISE:
    for sn in SPECKLE_NOISE:
        results = {'True Volume': [],
                   'Segmented Volume': [],
                   'Volume Absolute Error': [],
                   'Volume Percentage Error': [],
                   'True Surface Area': [],
                   'Segmented Surface Area': [],
                   'Surface Area Absolute Error': [],
                   'Surface Area Percentage Error': [],
                   'Root Mean Square Offset Error': []}
        suffix = 'gn_{:2.1f}_sn_{:2.1f}'.format(gn, sn).replace('.', 'p')
        for rep in range(REPEATS):
            out_directory = os.path.join(
                root_directory,
                '_'.join([suffix, '{:03d}'.format(rep + 1)]))
            root, all_seeds = generate_images.generate_test_images(a=7.0,
                                                                   b=3.5,
                                                                   c=2.5,
                                                                   background_noise=gn,
                                                                   speckle_noise=sn,
                                                                   spacing=0.2,
                                                                   number=1,
                                                                   deformed=10,
                                                                   output=out_directory)
            seg = segmenter.segmenter(image_directory=os.path.join(root, "reference"),
                                      spacing=[0.2, 0.2, 0.2],
                                      seed_points=all_seeds['reference'][0],
                                      bounding_box=[100, 100, 100],
                                      curvature_weight=10.0,
                                      area_weight=50.0,
                                      levelset_smoothing_radius=0.0,
                                      equalization_fraction=0.0)
            seg.execute()
            compare = DataComparison(os.path.join(seg.image_directory, 'ref.vtk'),
                                     seg.isocontours[-1])

            compare.write_polydata(os.path.join('_'.join([seg.image_directory, 'results']),
                                                'comparison'))
            results = update_results(results, compare)

            deformed_segmentations = []
            for deformation_iter in range(10):
                directory = os.path.join(root, "def_{:03d}".format(deformation_iter + 1))
                seg = segmenter.segmenter(image_directory=directory,
                                          spacing=[0.2, 0.2, 0.2],
                                          seed_points=all_seeds["deformed"][deformation_iter],
                                          bounding_box=[100, 100, 100],
                                          curvature_weight=10.0,
                                          area_weight=50.0,
                                          levelset_smoothing_radius=0.0,
                                          equalization_fraction=0.0)
                deformed_segmentations.append(seg)
                deformed_segmentations[-1].execute()
                compare = DataComparison(os.path.join(directory, "def.vtk"),
                                         deformed_segmentations[-1].isocontours[-1])
                compare.write_polydata(os.path.join('_'.join([directory, "results"]), "comparison"))

                results = update_results(results, compare)

        df = pandas.DataFrame(results)
        df.to_excel(os.path.join(root_directory, '.'.join([suffix, 'xlsx'])))
