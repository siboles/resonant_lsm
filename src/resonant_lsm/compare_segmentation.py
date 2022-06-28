import argparse
from pathlib import PurePath
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


class DataComparison:
    def __init__(self, surface1: vtk.vtkPolyData, surface2: vtk.vtkPolyData):
        self.surface1 = surface1
        self.surface2 = surface2
        self.surface1_volume, self.surface1_surface_area = self.get_volume_and_surface_area(self.surface1)
        self.surface2_volume, self.surface2_surface_area = self.get_volume_and_surface_area(self.surface2)

        self.error_polydata = self.get_shape_offset_error()
        self.shape_offset_rms_error = self.get_rms_error(self.error_polydata.GetPointData().GetScalars())

    @staticmethod
    def get_volume_and_surface_area(surface: vtk.vtkPolyData) -> vtk.vtkPolyData:
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(surface)
        mass_properties.Update()
        return mass_properties.GetVolume(), mass_properties.GetSurfaceArea()

    def get_shape_offset_error(self) -> vtk.vtkPolyData:
        distance = vtk.vtkDistancePolyDataFilter()
        distance.SetInputData(0, self.surface1)
        distance.SetInputData(1, self.surface2)
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


def read_vtk_polydata(filepath):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    return reader.GetOutput()


def main(surface_path_1, surface_path_2):
    surface1 = read_vtk_polydata(surface_path_1)
    surface2 = read_vtk_polydata(surface_path_2)
    comparison = DataComparison(surface1, surface2)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two segmented surfaces of the same object."
    )
    parser.add_argument("surface_path_1", help="Path to surface 1", type=str)
    parser.add_argument("surface_path_2", help="Path to surface 2", type=str)
    parser.add_argument("--output_name",
                        help="Output file name.", type=str, default="comparison")
    args = parser.parse_args()
    comparison = main(args.surface_path_1, args.surface_path_2, args.output_name)
    comparison.write_polydata(args.output_name)