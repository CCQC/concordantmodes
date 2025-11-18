import os
from concordantmodes.symmetry import Symmetry
from concordantmodes.options import Options
from concordantmodes.zmat import Zmat
from concordantmodes.s_vectors import SVectors

# import molsym
import numpy as np


def test_symmetry():
    os.chdir("./ref_data/symmetry_test/")
    test_subjects_red = ["c2v_water_zmat_red", "c3v_ammonia_zmat_red"]
    test_subjects_custom = ["c2v_water_zmat_custom", "c3v_ammonia_zmat_custom"]
    # redundants first
    test_subject_projs_red = [
        np.array(
            [
                [0.56086644, 0.43061449, -0.70710678],
                [0.56086644, 0.43061449, 0.70710678],
                [-0.60898085, 0.79318492, 0.0],
            ]
        ),
        np.array(
            [
                [-0.14852746, 0.55791839, 0.3291588, -0.24149772],
                [-0.14852746, 0.55791839, 0.3291588, -0.24149772],
                [-0.14852746, 0.55791839, -0.6583176, 0.48299545],
                [0.23900037, 0.06362601, -0.42112908, -0.5739944],
                [0.23900037, 0.06362601, 0.21056454, 0.2869972],
                [0.23900037, 0.06362601, 0.21056454, 0.2869972],
                [0.50413466, 0.13420931, 0.23650561, 0.3223546],
                [-0.50413466, -0.13420931, 0.1182528, 0.1611773],
                [0.50413466, 0.13420931, -0.1182528, -0.1611773],
            ]
        ),
    ]

    # start of customs
    test_subject_projs_custom = [
        np.array(
            [
                [0.56086644, 0.43061449, -0.70710678],
                [0.56086644, 0.43061449, 0.70710678],
                [-0.60898085, 0.79318492, 0.0],
            ]
        ),
        np.array(
            [
                [-0.14852746, 0.55791839, 0.3291588, -0.24149772],
                [-0.14852746, 0.55791839, 0.3291588, -0.24149772],
                [-0.14852746, 0.55791839, -0.6583176, 0.48299545],
                [0.23900037, 0.06362601, -0.42112908, -0.5739944],
                [0.23900037, 0.06362601, 0.21056454, 0.2869972],
                [0.23900037, 0.06362601, 0.21056454, 0.2869972],
                [0.50413466, 0.13420931, -0.1182528, -0.1611773],
                [0.50413466, 0.13420931, -0.1182528, -0.1611773],
                [0.50413466, 0.13420931, 0.23650561, 0.3223546],
            ]
        ),
    ]
    # iterate over redundant and custom coordinates
    for c, coord in enumerate(["redundant", "Custom"]):
        if coord == "redundant":
            test_subjects = test_subjects_red
            test_subject_projs = test_subject_projs_red
        else:
            test_subjects = test_subjects_custom
            test_subject_projs = test_subject_projs_custom

        # iterate over zmats to test symmetry projection on
        for t, test in enumerate(test_subjects):
            options = Options()
            options.cart_insert = 9
            options.molsym_symmetry = True
            options.coords = coord
            options.second_order = False
            zmat = Zmat(options)
            zmat.run(test)
            symm_obj = Symmetry(zmat, options)
            symm_obj.run()
            s_vec = SVectors(zmat, options, zmat.variable_dictionary_init)
            s_vec.run(zmat.cartesians_init, True, second_order=options.second_order)
            symm_obj.make_proj(s_vec)
            assert np.allclose(
                symm_obj.salc_proj, test_subject_projs[t], rtol=0, atol=1e-8
            )
