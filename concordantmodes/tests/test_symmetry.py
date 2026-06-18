import os
from concordantmodes.symmetry import Symmetry
from concordantmodes.options import Options
from concordantmodes.zmat import Zmat
from concordantmodes.s_vectors import SVectors

# import molsym
import numpy as np


def test_symmetry():
    os.chdir("./ref_data/symmetry_test/")
    test_subjects_del = ["c2v_water_zmat_del", "c3v_ammonia_zmat_del"]
    test_subjects_custom = ["c2v_water_zmat_custom", "c3v_ammonia_zmat_custom"]
    # delocalized first
    test_subject_projs_del = [
        np.array(
            [
                [-0.56086644, 0.43061449, -0.70710678],
                [-0.56086644, 0.43061449, 0.70710678],
                [0.60898085, 0.79318492, 0.0],
            ]
        ),
        np.array(
            [
                [-0.14852746, -0.55791839, -0.3291588 , -0.57011977, -0.24149772,  0.41828633], 
                [-0.14852746, -0.55791839, -0.3291588 ,  0.57011977, -0.24149772, -0.41828633],
                [-0.14852746, -0.55791839,  0.6583176 ,  0.        ,  0.48299545, -0.        ],
                [ 0.23900037, -0.06362601,  0.42112908,  0.        , -0.5739944 ,  0.        ],
                [ 0.23900037, -0.06362601, -0.21056454,  0.36470848,  0.2869972 ,  0.49709374],
                [ 0.23900037, -0.06362601, -0.21056454, -0.36470848,  0.2869972 , -0.49709374],
                [ 0.50413466, -0.13420931, -0.23650561,  0.        ,  0.3223546 ,  0.        ],
                [-0.50413466,  0.13420931, -0.1182528 ,  0.20481986,  0.1611773 ,  0.27916727],
                [ 0.50413466, -0.13420931,  0.1182528 ,  0.20481986, -0.1611773 ,  0.27916727]
            
            ]
        ),
    ]

    # start of customs
    test_subject_projs_custom = [
        np.array(
            [
                [-0.56086644, 0.43061449, -0.70710678],
                [-0.56086644, 0.43061449, 0.70710678],
                [0.60898085, 0.79318492, 0.0],
            ]
        ),
        np.array(
            [
                [-0.14852746, -0.55791839, -0.3291588 ,  0.57011977, -0.24149772,  0.41828633], 
                [-0.14852746, -0.55791839, -0.3291588 , -0.57011977, -0.24149772, -0.41828633],
                [-0.14852746, -0.55791839,  0.6583176 , -0.        ,  0.48299545, -0.        ],
                [ 0.23900037, -0.06362601,  0.42112908,  0.        , -0.5739944 ,  0.        ],
                [ 0.23900037, -0.06362601, -0.21056454,  0.36470848,  0.2869972 , -0.49709374],
                [ 0.23900037, -0.06362601, -0.21056454, -0.36470848,  0.2869972 ,  0.49709374],
                [ 0.50413466, -0.13420931,  0.1182528 , -0.20481986, -0.1611773 ,  0.27916727],
                [ 0.50413466, -0.13420931,  0.1182528 ,  0.20481986, -0.1611773 , -0.27916727],
                [ 0.50413466, -0.13420931, -0.23650561,  0.        ,  0.3223546 , -0.        ] 
          
            ]
        ),
    ]
    # iterate over delocalized and custom coordinates
    for c, coord in enumerate(["Delocalized", "Custom"]):
        if coord == "Delocalized":
            test_subjects = test_subjects_del
            test_subject_projs = test_subject_projs_del
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
            symm_obj = Symmetry(zmat, options, test_subject_projs)
            symm_obj.run()
            s_vec = SVectors(zmat, options)
            s_vec.run(zmat.cartesians_b, True, second_order=options.second_order)
            symm_obj.make_proj(s_vec)
            assert np.allclose(
                symm_obj.salc_proj, test_subject_projs[t], rtol=0, atol=1e-8
            )
    os.chdir("../../")
