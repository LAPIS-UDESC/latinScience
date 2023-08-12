class GetDatasetParams:

    def get_dataset_params(dataset_path: str):

        # dataset metadate collection
        temp_str = dataset_path.split("/")[-1]
        temp_str = temp_str.split(".")[0]
        temp_str = temp_str.split("|")
        dataset_params = dict(
            {"n_samples": int(temp_str[-1].replace("n_samples=", ""))})

        if(temp_str[0] == "linearized"):
            dataset_params.update({"type": temp_str[0]})
        elif(temp_str[0] == "lbp" or temp_str[0] == "lbp_rs" or "lbp_BND"):
            dataset_params.update({"type": temp_str[0],
                                "radius": int(temp_str[1].replace("radius=", "")),
                                "n_points": int(temp_str[2].replace("n_points=", ""))})
        else:
            dataset_params.update({"type": temp_str[0],
                                "orientations": int(temp_str[1].replace("orientations=", "")),
                                "pixels_per_cell": temp_str[2].replace("pixels_per_cell=", ""),
                                "cells_per_block": temp_str[3].replace("cells_per_block=", ""),
                                "block_norm": temp_str[4].replace("block_norm=", ""),
                                })

            feature_vector = ""
            if temp_str[5] == "True":
                feature_vector = True
            else:
                feature_vector = False

            transform_sqrt = ""
            if temp_str[6] == "True":
                transform_sqrt = True
            else:
                transform_sqrt = False

            dataset_params.update({"feature_vector": feature_vector,
                                "transform_sqrt": transform_sqrt,
                                "channel_axis": temp_str[7]})

        return dataset_params