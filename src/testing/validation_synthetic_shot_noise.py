import generateImages
from resonant_lsm import segmenter

# gaussian_noise_levels = [0.0, 0.1, 0.2]
# shot_noise_levels = [0.0, 0.1, 0.2, 0.3]
gaussian_noise_levels = [0.1]
shot_noise_levels = [0.0]
for gn in gaussian_noise_levels:
    for sn in shot_noise_levels:
        out_directory = 'gn_{:2.1f}_sn_{:2.1f}'.format(gn, sn).replace('.', 'p')
        root, all_seeds = generateImages.generate_test_images(background_noise=gn,
                                                              shot_noise=sn,
                                                              number=1,
                                                              deformed=0,
                                                              output=out_directory)
        print(all_seeds['reference'][0])
        seg = segmenter(image_directory=root,
                        spacing=[0.1, 0.1, 0.1],
                        seed_points=all_seeds['reference'][0],
                        bounding_box=[100, 100, 100],
                        levelset_smoothing_scale=0.0,
                        equalization_fraction=0.0)
        seg.execute()
        seg.write_image_as_vtk(seg.levelsets[-1], 'levelset')
