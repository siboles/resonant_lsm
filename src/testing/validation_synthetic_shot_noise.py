from resonant_lsm import segmenter, generate_images

gaussian_noise_levels = [0.0, 0.1, 0.2]
shot_noise_levels = [0.0, 0.1, 0.2]
for gn in gaussian_noise_levels:
    for sn in shot_noise_levels:
        out_directory = 'gn_{:2.1f}_sn_{:2.1f}'.format(gn, sn).replace('.', 'p')
        root, all_seeds = generate_images.generate_test_images(background_noise=gn,
                                                               speckle_noise=sn,
                                                               spacing=0.1,
                                                               number=1,
                                                               deformed=0,
                                                               output=out_directory)
        seg = segmenter(image_directory=root,
                        spacing=[0.1, 0.1, 0.1],
                        seed_points=all_seeds['reference'][0],
                        bounding_box=[100, 100, 100],
                        curvature_weight=0.0,
                        area_weight=50.0,
                        levelset_smoothing_radius=0.0,
                        equalization_fraction=0.0)
        seg.execute()
