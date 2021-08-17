import os
import argparse

import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np


def _generate_super_ellipsoid(a, b, c, n1, n2):
    super_ellipsoid = vtk.vtkParametricSuperEllipsoid()
    super_ellipsoid.SetXRadius(a)
    super_ellipsoid.SetYRadius(b)
    super_ellipsoid.SetZRadius(c)
    super_ellipsoid.SetN1(n1)
    super_ellipsoid.SetN2(n2)

    ratios = np.array([a, b, c])
    ratios = ratios / np.max(ratios)
    super_ellipsoid_source = vtk.vtkParametricFunctionSource()
    super_ellipsoid_source.SetParametricFunction(super_ellipsoid)
    super_ellipsoid_source.SetUResolution(np.ceil(100 * ratios[0]).astype(int))
    super_ellipsoid_source.SetVResolution(np.ceil(100 * ratios[1]).astype(int))
    super_ellipsoid_source.SetWResolution(np.ceil(100 * ratios[2]).astype(int))

    super_ellipsoid_source.Update()
    polydata = super_ellipsoid_source.GetOutput()
    return polydata


def _pack_objects(objects, spacing):
    bb = np.array([p.GetBounds() for p in objects])
    bb[:, 0::2] -= np.array([spacing] * 3)
    bb[:, 1::2] += np.array([spacing] * 3)
    dimensions = np.zeros((bb.shape[0], 3), dtype=float)
    dimensions[:, 0] = bb[:, 1] - bb[:, 0]
    dimensions[:, 1] = bb[:, 3] - bb[:, 2]
    dimensions[:, 2] = bb[:, 5] - bb[:, 4]
    # sort first by height
    order = np.argsort(dimensions[:, 2])
    # sort sub-groups of 4 minimizing total surface area
    for i in np.arange(0, order.size, 4):
        remainder = order.size - i
        if remainder > 4:
            i_1 = i + 1
            i_2 = i + 2
            i_3 = i + 3
            i_4 = i + 4
            sortix = np.argsort(dimensions[order[i: i_4], 0])
            order[i:i_4] = order[i:i_4][sortix]
            # swap middle two if y lengths sum to less
            sum1 = dimensions[order[i], 1] + dimensions[order[i_1], 1]
            sum2 = dimensions[order[i_2], 1] + dimensions[order[i_3], 1]
            sum3 = dimensions[order[i], 1] + dimensions[order[i_2], 1]
            sum4 = dimensions[order[i_1], 1] + dimensions[order[i_3], 1]
            if max([sum1, sum2]) > max([sum3, sum4]):
                order[i:i_4] = order[i:i_4][[0, 2, 1, 3]]
        else:
            sortix = np.argsort(dimensions[order[i::], 0])
            order[i::] = order[i::][sortix]

    scene = vtk.vtkAppendPolyData()
    h = 0
    cnt = 0
    for i in np.arange(0, order.size, 4):
        remainder = order.size - i
        if remainder > 4:
            ind = order[i:i + 4]
        else:
            ind = order[i::]
        h += np.max(dimensions[ind, 2])
        for j in range(ind.size):
            cnt += 1
            tx = vtk.vtkTransform()
            mz = h - bb[ind[j], 5]
            if j == 0:
                mx = bb[ind[j], 0]
                my = bb[ind[j], 3]
            elif j == 1:
                mx = bb[ind[j], 0]
                my = bb[ind[j], 2]
            elif j == 2:
                mx = bb[ind[j], 1]
                my = bb[ind[j], 2]
            else:
                mx = bb[ind[j], 1]
                my = bb[ind[j], 3]
            tx.Translate([mx, my, mz])
            tx_filter = vtk.vtkTransformPolyDataFilter()
            tx_filter.SetInputData(objects[ind[j]])
            tx_filter.SetTransform(tx)
            scene.AddInputConnection(tx_filter.GetOutputPort())
            scene.Update()
    return scene.GetOutput()


def _deform_poly_data(p, spacing, scale):
    # create thin-plate spline control po0.05ints
    bounds = p.GetBounds()
    step_size = np.array([spacing] * 3) * 10
    div = [np.ceil(np.abs(bounds[2 * i + 1] - bounds[2 * i]) / step_size[i]).astype(int) for i in range(3)]
    x, x_step = np.linspace(bounds[0], bounds[1], div[0], retstep=True)
    y, y_step = np.linspace(bounds[2], bounds[3], div[1], retstep=True)
    z, z_step = np.linspace(bounds[4], bounds[5], div[2], retstep=True)
    source_points = np.meshgrid(x, y, z)
    x_perturb = np.random.normal(loc=0.0, scale=scale * x_step, size=source_points[0].size)
    y_perturb = np.random.normal(loc=0.0, scale=scale * y_step, size=source_points[1].size)
    z_perturb = np.random.normal(loc=0.0, scale=scale * z_step, size=source_points[2].size)

    all_source_points = np.zeros(source_points[0].size * 3)
    all_source_points[0::3] = source_points[0].ravel()
    all_source_points[1::3] = source_points[1].ravel()
    all_source_points[2::3] = source_points[2].ravel()

    all_target_points = np.copy(all_source_points)
    all_target_points[0::3] += x_perturb
    all_target_points[1::3] += y_perturb
    all_target_points[2::3] += z_perturb

    source_points = vtk.vtkPoints()
    target_points = vtk.vtkPoints()
    arr1 = numpy_support.numpy_to_vtk(all_source_points, deep=True, array_type=vtk.VTK_DOUBLE)
    arr1.SetNumberOfComponents(3)
    arr2 = numpy_support.numpy_to_vtk(all_target_points, deep=True, array_type=vtk.VTK_DOUBLE)
    arr2.SetNumberOfComponents(3)
    source_points.SetData(arr1)
    target_points.SetData(arr2)

    transform = vtk.vtkThinPlateSplineTransform()
    transform.SetSourceLandmarks(source_points)
    transform.SetTargetLandmarks(target_points)
    transform.SetBasisToR()

    poly_transform = vtk.vtkTransformPolyDataFilter()
    poly_transform.SetInputData(p)
    poly_transform.SetTransform(transform)
    poly_transform.Update()

    polydata = poly_transform.GetOutput()
    return polydata


def _poly2img(p, spacing, object_noise, background_noise):
    bb = np.array(p.GetBounds())
    extent = [np.ceil(np.abs(bb[2 * i + 1] - bb[2 * i]) / spacing).astype(int) + 10 for i in range(3)]

    arr = numpy_support.numpy_to_vtk(np.ones(extent, np.float32).ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    arr.SetName('Intensity')
    arr.SetNumberOfComponents(1)

    img = vtk.vtkImageData()
    img.SetSpacing([spacing] * 3)
    img.SetExtent((0, extent[0] - 1, 0, extent[1] - 1, 0, extent[2] - 1))
    img.SetOrigin([bb[0] - 5 * spacing,
                   bb[2] - 5 * spacing,
                   bb[4] - 5 * spacing])
    img.GetPointData().SetScalars(arr)
    p2im = vtk.vtkPolyDataToImageStencil()
    p2im.SetInputData(p)
    p2im.SetOutputOrigin(img.GetOrigin())
    p2im.SetOutputSpacing(img.GetSpacing())
    p2im.SetOutputWholeExtent(img.GetExtent())
    p2im.SetTolerance(spacing / 10.0)
    p2im.Update()

    image_stencil = vtk.vtkImageStencil()
    image_stencil.SetInputData(img)
    image_stencil.SetStencilConnection(p2im.GetOutputPort())
    image_stencil.ReverseStencilOff()
    image_stencil.SetBackgroundValue(0.0)
    image_stencil.Update()
    img = image_stencil.GetOutput()

    arr = numpy_support.vtk_to_numpy(img.GetPointData().GetArray('Intensity')).reshape(extent[2], extent[1], extent[0])
    itk_img = sitk.GetImageFromArray(arr)
    itk_img.SetSpacing(img.GetSpacing())
    itk_img.SetOrigin(img.GetOrigin())

    mask = sitk.BinaryThreshold(itk_img, 0.5, 1e3)

    # object_noise_img = sitk.AdditiveGaussianNoise(itk_img, standardDeviation=object_noise)
    # object_noise_img = sitk.RescaleIntensity(object_noise_img, 0.0, 1.0) * itk_img
    # background_noise_img = sitk.AdditiveGaussianNoise(itk_img,
    #                                                   standardDeviation=background_noise) * sitk.InvertIntensity(
    #    itk_img, maximum=1.0)
    # background_noise_img = sitk.RescaleIntensity(background_noise_img, 0.0, background_noise)
    # itk_img = object_noise_img + background_noise_img
    itk_img = sitk.ShotNoise(itk_img, scale=object_noise)
    itk_img = sitk.RescaleIntensity(itk_img, 0.0, 1.0)
    labels = sitk.ConnectedComponent(mask)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(labels)
    regions = []
    for label in ls.GetLabels():
        bb = ls.GetBoundingBox(label)
        origin = [i - 2 for i in bb[0:3]]
        size = [i + 4 for i in bb[3:]]
        regions.append(origin + size)
    return itk_img, regions


def generate_test_images(a=2.0, b=1.0, c=1.0, n1=0.9, n2=0.9, spacing=0.1, output=None, number=1,
                         deformed=0, object_noise=0.3, background_noise=0.1):
    """
    Description
    -----------
    Generates 3D image(s) of cell-like geometry. Optionally, generate reference and deformed pairs. If number > 1
    then radii are varied +/- 15%.

    Parameters
    ----------
    a : float=2.0
       The x-radius of the super-ellipsoid.
    b : float=1.0
       The y-radius of the super-ellipsoid.
    c : float=1.0
       The z-radius of the super-ellipsoid.
    n1 : float 0.9
       Shape parameter in v; (0.0, 1.0] ranges from squared to ellipsoidal corners, > 1.0 concave surface with sharp
       edges.
    n2 : float 0.9
       Shape parameter in u; (0.0, 1.0] ranges from squared to ellipsoidal corners, > 1.0 concave surface with sharp
       edges.
    spacing : float 0.1
       Voxel edge length
    output : str
       The output directory to write image(s) to. If not provided the current working directory will be used.
    number : int=1
       Number of reference images to generate.
    deformed : int=0
       Number of deformed images to create from each reference image.
    object_noise : float=0.3
       Standard deviation of Gaussian noise to add to objects in images.
    background_noise : float=0.1
       Standard deviation of Gaussian noise to add to background of images.

    Returns
    -------
    root : str
       Root directory for output images.
    regions : dict
       Dictionary containing 'reference' and 'deformed' regions of interest of for each cell.
    """
    if output is None:
        root = os.getcwd()
    elif os.path.isabs(output):
        root = output
    else:
        root = os.path.join(os.getcwd(), output)

    if not os.path.exists(root):
        os.mkdir(root)

    if number > 1:
        a = np.random.uniform(low=0.85 * a, high=1.15 * a, size=number)
        b = np.random.uniform(low=0.85 * b, high=1.15 * b, size=number)
        c = np.random.uniform(low=0.85 * c, high=1.15 * c, size=number)
    else:
        a = [a]
        b = [b]
        c = [c]

    objects = []
    for i in range(number):
        objects.append(_generate_super_ellipsoid(a[i], b[i], c[i], n1, n2))

    polydata = _pack_objects(objects, spacing)
    ref_polydata = _deform_poly_data(polydata, spacing, 0.1)
    ref_img, regions = _poly2img(ref_polydata, spacing, object_noise, background_noise)

    all_regions = {"reference": [regions], "deformed": []}
    sitk.WriteImage(ref_img, os.path.join(root, "ref.nii"))
    for i in range(deformed):
        def_polydata = _deform_poly_data(ref_polydata, spacing, 0.1)
        def_img, regions = _poly2img(def_polydata, spacing, object_noise, background_noise)
        all_regions["deformed"].append(regions)
        sitk.WriteImage(def_img, os.path.join(root, "def_{:03d}.nii".format(i + 1)))
    return root, all_regions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates 3D image(s) of cell-like geometry. Optionally, generate reference and deformed pairs. '
                    'If number > 1 then radii are varied +/- 15%.')
    parser.add_argument('-a', type=float, default=2.0, help='float : x-radius of base super-ellipsoid')
    parser.add_argument('-b', type=float, default=1.0, help='float : y-radius of base super-ellipsoid')
    parser.add_argument('-c', type=float, default=1.0, help='float : z-radius of base super-ellipsoid')
    parser.add_argument('-n1', type=float, default=0.9,
                        help='float : shape parameter in v; (0.0,1.0) square to rounded corners '
                             '1.0 is ellipsoid, > 1.0 concave with sharp edges')
    parser.add_argument('-n2', type=float, default=0.9,
                        help='float : shape parameter in u; (0.0,1.0) square to rounded corners '
                             '1.0 is ellipsoid, > 1.0 concave with sharp edges')
    parser.add_argument('-spacing', type=float, default=0.1)
    parser.add_argument('-output', type=str, default=None, help='str : output directory to write images to')
    parser.add_argument('-number', type=int, default=1, help='int : how many clustered cells to generate')
    parser.add_argument('-deformed', type=int, default=0, help='int : how many deformed images to generate.')
    parser.add_argument('-object_noise', type=float, default=0.3,
                        help='float : standard deviation of Gaussian noise to add to objects in images.')
    parser.add_argument('-background_noise', type=float, default=0.1,
                        help='float : standard deviation of Gaussian noise to add to background of images.')
    args = parser.parse_args()
    generate_test_images(a=args.a, b=args.b, c=args.c, n1=args.n1, n2=args.n2, spacing=args.spacing, output=args.output,
                         deformed=args.deformed, number=args.number, object_noise=args.object_noise,
                         background_noise=args.background_noise)