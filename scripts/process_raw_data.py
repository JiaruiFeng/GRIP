import os

from data import *

OUTPUT_DIR = "outputs/data/"

DATASET_DICT = {"scene_graph": process_scene_graph,
                "fb15k237_2": process_fb15k237_2,
                "wn18rr": process_wn18rr,
                "clegr": process_clegr,
                "nell23k": process_nell23k,
                "codexm": process_codexm,
                }

if __name__ == "__main__":
    for dataset, process_func in DATASET_DICT.items():
        output_path = OUTPUT_DIR + dataset
        if (not os.path.exists(output_path + "/processed_test.json")):
            print(f"Processing {dataset} dataset...")
            process_func(output_path)
