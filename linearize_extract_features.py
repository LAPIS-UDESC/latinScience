import pickle
import shutil
from os import mkdir

import cv2
from genericpath import exists
from joblib import Parallel, delayed

from modules.to_gray import toGray


def build_dataset(db_name, n_samples, force_flag):

    # convert images to greyscale
    gray = toGray()

    # dataset path
    dataset_path = "../../datasets"

    # dataset name
    name = "linearized|n_samples="+str(n_samples)

 # extraction task
    def task(i):

        path_covid = "../../database/"+db_name + \
            "/COVID/images/COVID-"+str(i)+".png"
        path_normal = "../../database/"+db_name + \
            "/Normal/images/Normal-"+str(i)+".png"

        print(path_covid, end="\r")
        print(path_normal, end="\r")

        img_covid = gray.image_to_gray(cv2.imread(path_covid))
        img_normal = gray.image_to_gray(cv2.imread(path_normal))

        pickle_covid = img_covid.ravel()
        pickle_normal = img_normal.ravel()

        with open(dataset_path+"/"+name+"/COVID/"+str(i), "xb") as f:
            pickle.dump(pickle_covid, f)

        with open(dataset_path+"/"+name+"/NORMAL/"+str(i), "xb") as f:
            pickle.dump(pickle_normal, f)

    # check if dataset already built
    if(not exists(dataset_path)):
        mkdir(dataset_path)

    if(not exists(dataset_path+"/"+name) or force_flag):

        if(exists(dataset_path+"/"+name)):
            print("rebuilding already built flattened dataset")
            shutil.rmtree(dataset_path+"/"+name)

        mkdir(dataset_path+"/"+name)
        mkdir(dataset_path+"/"+name+"/"+"COVID")
        mkdir(dataset_path+"/"+name+"/"+"NORMAL")

        Parallel(n_jobs=16)(delayed(task)(i) for i in range(1, n_samples+1))

        print("\n")
        print("dataset "+name+" built!")
    else:
        print("skipping already built linearized dataset")
