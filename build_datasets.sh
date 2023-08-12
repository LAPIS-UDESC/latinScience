# !/bin/bash

#dataset build

echo "building all datasets from scratch"

#python3 dataset_build_utility.py -d COVID-19_Radiography_Dataset -n 3616 -e linearize -f
#python3 dataset_build_utility.py -d COVID-19_Radiography_Dataset -n 3616 -e lbp -a /home/caloinoggins/workspace/ic_rough_sets/args/lbp_arguments.json -f
#python3 dataset_build_utility.py -d COVID-19_Radiography_Dataset -n 3616 -e hog -a /home/caloinoggins/workspace/ic_rough_sets/args/hog8x8.json  -f 
#python3 dataset_build_utility.py -d COVID-19_Radiography_Dataset -n 3616 -e hog -a /home/caloinoggins/workspace/ic_rough_sets/args/hog16x16.json -f 

#python3 dataset_build_utility.py -d augmented -n 10000 -e linearize -f
#python3 dataset_build_utility.py -d augmented -n 10000 -e lbp -a  /home/caloinoggins/workspace/ic_rough_sets/args/lbp_arguments.json -f
#python3 dataset_build_utility.py -d augmented -n 10000 -e hog -a /home/caloinoggins/workspace/ic_rough_sets/args/hog8x8.json -f
#python3 dataset_build_utility.py -d augmented -n 10000 -e hog -a /home/caloinoggins/workspace/ic_rough_sets/args/hog16x16.json -f

python3 dataset_build_utility.py -d bimcv_kaggle -n 10192 -e lbp -a /home/caloinoggins/workspace/ic_rough_sets/args/lbp_arguments.json -f
#python3 dataset_build_utility.py -d bimcv_kaggle -n 10192 -e hog -a /home/caloinoggins/workspace/ic_rough_sets/args/hog8x8.json -f
#python3 dataset_build_utility.py -d bimcv_kaggle -n 10192 -e hog -a /home/caloinoggins/workspace/ic_rough_sets/args/hog8x8.json -f
#python3 dataset_build_utility.py -d bimcv_kaggle -n 10192 -e hog -a /home/caloinoggins/workspace/ic_rough_sets/args/hog8x8.json -f

echo "datasets built! all done!"
