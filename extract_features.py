import shutil
from os import mkdir, path, remove, walk

import cv2
import numpy as np
import tables
import json
from sklearn.decomposition import PCA

from genericpath import exists
from joblib import Parallel, delayed
from modules.to_gray import toGray

# Load PATH variable
with open("config.json", "r") as file:
    config = json.load(file)
DATASETS = config["paths"]["DATASETS"]
DATABASES = config["paths"]["DATABASES"]
SOURCE = config["paths"]["SOURCE"]

# define number of threads to run extraction task. -1 to use all threads
NTHREADS = -1


def build_dataset(
    descriptor_instance, db_name, n_samples, force_flag, type, pca_flag, normalize
):
    # convert images to greyscale
    gray = toGray()

    # dataset build path
    dataset_build_path = path.join(DATASETS, "build")

    # dataset path
    dataset_path = DATASETS

    # dataset name
    name = ""

    # get metadata of dataset
    metadata = dict(
        {
            "n_samples": str(n_samples),
            "pca": False,
            "normalize": False,
            "descriptor": type,
        }
    )

    if normalize:
        metadata.update({"normalize": True})

    # select dataset name based on type and prepare
    # preapare dict with metadata information about the dataset
    if type == "lbp":
        # update with descriptor metadata
        metadata.update(
            {
                "descriptor_metadata": {
                    "radius": str(descriptor_instance.n_points),
                    "n_points": str(descriptor_instance.radius),
                },
            }
        )

        # dataset name with current lbp parameters
        name = "lbp"
        if pca_flag:
            name += "_pca"
            metadata.update({"pca": True})

        name += (
            "|radius="
            + str(descriptor_instance.radius)
            + "|n_points="
            + str(descriptor_instance.n_points)
            + "|n_samples="
            + str(n_samples)
        )

    # check if directory structure exist. If not, make it
    if not exists(dataset_path):
        mkdir(dataset_path)

    if not exists(path.join(dataset_path, db_name)):
        mkdir(path.join(dataset_path, db_name))

    if not exists(path.join(dataset_path, db_name, type)):
        mkdir(path.join(dataset_path, db_name, type))

    if not exists(path.join(dataset_path, db_name, type, "normalizado")):
        mkdir(path.join(dataset_path, db_name, type, "normalizado"))

    if not exists(path.join(dataset_path, db_name, type, "nao_normalizado")):
        mkdir(path.join(dataset_path, db_name, type, "nao_normalizado"))

    # check if dataset has ben built. if not or if force flag, build it
    if not exists(path.join(dataset_path, db_name, type, name)) or force_flag:
        if exists(dataset_path + "/" + name):
            print(
                "rebuilding already built dataset with paramaters:"
                + " radius="
                + str(descriptor_instance.radius)
                + " n_points="
                + str(descriptor_instance.n_points)
                + " images="
                + str(n_samples)
            )
            remove(path.join(dataset_path, db_name, type, name))

        # make shute build directory is clean
        if exists(dataset_build_path):
            shutil.rmtree(dataset_build_path)

        # make build directory
        mkdir(dataset_build_path)
        mkdir(dataset_build_path + "/" + name)
        mkdir(dataset_build_path + "/" + name + "/" + "COVID")
        mkdir(dataset_build_path + "/" + name + "/" + "NORMAL")

        # spawn extraction tasks to calculate descriptors in parallel
        Parallel(n_jobs=NTHREADS)(
            delayed(extraction_task)(
                i,
                db_name,
                gray,
                descriptor_instance,
                name,
                dataset_build_path,
                normalize,
            )
            for i in range(1, n_samples + 1)
        )

        print(metadata)
        # join files
        join_files(
            dataset_build_path,
            dataset_path,
            db_name,
            type,
            name,
            n_samples,
            pca_flag,
            normalize,
        )

        # clean build directory tree
        shutil.rmtree(dataset_build_path)

        print("\n")
        print("dataset " + name + " built!")
    else:
        print("skipping already built dataset with paramaters:\n" + name)


# extraction task
def extraction_task(
    i, db_name, gray, descriptor_instance, name, dataset_build_path, normalize
):
    path_covid = DATABASES + db_name + "/COVID/COVID-" + str(i) + ".png"
    path_normal = DATABASES + db_name + "/Normal/Normal-" + str(i) + ".png"

    print(path_covid, end="\r")
    print(path_normal, end="\r")

    # open image and grayscale it
    img_covid = gray.image_to_gray(cv2.imread(path_covid))
    img_normal = gray.image_to_gray(cv2.imread(path_normal))

    # calculate descriptors
    descriptor_covid = np.ravel(descriptor_instance.describe(img_covid))
    descriptor_normal = np.ravel(descriptor_instance.describe(img_normal))

    # normalize if flag is set to True
    if normalize:
        descriptor_covid = descriptor_covid.astype("float")
        descriptor_covid /= descriptor_covid.max()
        descriptor_normal = descriptor_normal.astype("float")
        descriptor_normal /= descriptor_normal.max()

    # save files on disk for later processing
    with open(dataset_build_path + "/" + name + "/COVID/" + str(i), "xb") as f:
        np.save(f, descriptor_covid)

    with open(dataset_build_path + "/" + name + "/NORMAL/" + str(i), "xb") as f:
        np.save(f, descriptor_normal)


# join individual descriptors for each class into a sigle compressed .npz array file
def join_files(
    dataset_build_path,
    dataset_path,
    db_name,
    type,
    name,
    n_samples,
    pca_flag,
    normalize,
):
    print("joining files")

    # create and open hdf5 table
    if normalize:
        hdf5_filename = path.join(
            dataset_path, db_name, type, "normalizado", name + ".hdf5"
        )
    else:
        hdf5_filename = path.join(
            dataset_path, db_name, type, "nao_normalizado", name + ".hdf5"
        )

    hdf5 = tables.open_file(hdf5_filename, "w")

    # checa se deve aplicar ou não redução de dimensionalidade com PCA
    if pca_flag:
        # instaciamento do pca
        pca = PCA(n_components="mle")

        descriptors = np.array()
        labels = np.array()

        # iterate throgh files
        for root, dirs, files in walk(dataset_build_path):
            for file in files:
                temp_path = path.join(root, file)
                category = temp_path.split("/")[-2]

                with open(temp_path, "rb") as f:
                    descriptor = np.load(f)
                    descriptors.append(descriptors)
                    labels.append(category)
                    del descriptor

        print(descriptors, descriptors.shape)
        descriptors = pca.fit_transform(descriptors)
        print(descriptors, descriptors.shape)

    else:
        # open sample file to get information to build pytables Earray
        with open(path.join(dataset_build_path, name, "COVID", str(1)), "rb") as f:
            tempfile = np.load(f)
            print(tempfile.dtype)

        n_rows = n_samples
        n_columns = len(tempfile)
        tempfile = np.expand_dims(tempfile, axis=0)

        # filtros para a tabela
        filters = tables.Filters(
            complevel=9, complib="blosc", shuffle=True, bitshuffle=True
        )

        # prototipo do dataset
        class Dataset(tables.IsDescription):
            label = tables.StringCol(itemsize=16)
            descriptor = tables.Float64Col(shape=(n_columns,))

        # criação da tabela
        table = hdf5.create_table(
            "/", "dataset", Dataset, filters=filters, expectedrows=2 * n_rows
        )

        # deleta arquivo temporario utilizado para ler dimensoes do descritor
        del tempfile

        # iterate throgh files
        for root, dirs, files in walk(dataset_build_path):
            for file in files:
                temp_path = path.join(root, file)
                category = temp_path.split("/")[-2]

                with open(temp_path, "rb") as f:
                    descriptor = np.load(f)
                    row = table.row
                    row["label"] = category
                    row["descriptor"] = descriptor
                    row.append()
                    del descriptor

                table.flush()
        # close file handler
    hdf5.close()
