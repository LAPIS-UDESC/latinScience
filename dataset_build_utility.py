import getopt
import json
import sys

import extract_features
import linearize_extract_features as linearize_extract
from modules.hog import HistogramOfOrientedGradients as HOG
from modules.lbp import LocalBinaryPatterns as LBP


def dataset_build_utility(argv):
    # full path and name to image database
    db_name = ""
    # npercent of samples taken from database
    n_samples = 0
    # feature extraction method
    extractor = ""
    # json with feature extractor arguments
    extractor_arguments = ""
    # force dataset build flag
    force_flag = False
    # pca flag
    pca_flag = False
    # normalize flag
    normalize = False

    try:
        options = "hd:n:e:a:pca:n:f"
        long_options = ["help", "database=", "n_sample=",
                        "extractor=", "extractor_arguments=","pca_flag=","normalize=", "force="]
        args, opts = getopt.getopt(argv, options, long_options)

        for current_arg, current_val in args:

            if(current_arg) in ("-h", "--help"):
                print("Feature extraction helper tool. Usage:")
                print("-h, --help                   Display this help dialog")
                print("-d, -database                Full path to database")
                print("-n, --n_sample               Number of images")
                print(
                    "-e,--extractor               Feature extraction method. Valid options: hog, lbp, linearize, tails")
                print("-a,--extractor-arguments     Full path to argument JSON. More information on https://github.com/kaloi-noggins/ic_rough_sets/")
                print("-pca,--pca_flag     Use PCA on dataset")
                print("-norm,--normalize     Normalize dataset")
                print("-f,--force                   Force dataset build")
            elif(current_arg) in ("-d", "--database"):
                db_name = current_val
            elif(current_arg) in ("-n", "--n_sample"):
                n_samples = int(current_val)
            elif(current_arg) in ("-e", "--extractor"):
                extractor = current_val
            elif(current_arg) in ("-a", "--extractor_arguments"):
                extractor_arguments = json.load(open(current_val))
            elif(current_arg) in ("-pca", "--pca_flag"):
                pca_flag = True
            elif(current_arg) in ("-norm", "--normalize"):
                normalize = True
            elif(current_arg) in ("-f", "--force"):
                force_flag = True
        
    except getopt.error as err:
        print(str(err))

    # extractor treatment
    if(extractor == "hog"):
        for i in extractor_arguments:
            orientations = i["orientations"]
            pixels_per_cell = tuple(map(int, i["pixels_per_cell"]))
            cells_per_block = tuple(map(int, i["cells_per_block"]))
            block_norm = i["block_norm"]
            transform_sqrt = i["transform_sqrt"]
            channel_axis = i["channel_axis"]

            if channel_axis == "None":
                channel_axis = None

            hog = HOG(orientations, pixels_per_cell, cells_per_block, block_norm,
                      transform_sqrt, channel_axis)

            extract_features.build_dataset(
                hog, db_name, n_samples, force_flag, "hog", pca_flag, normalize)

    elif(extractor == "lbp"):
        for i in extractor_arguments:
            radius = i["radius"]
            n_points = i["n_points"]*radius

            lbp = LBP(n_points, radius)
            extract_features.build_dataset(
                lbp, db_name, n_samples, force_flag, "lbp", pca_flag, normalize)

    elif(extractor == "linearize"):
        linearize_extract.build_dataset(db_name, n_samples, force_flag)


dataset_build_utility(sys.argv[1:])
