# from scripts.benchmarking.monusac_segmentation_evaluation import evaluate_segmentation_files
from monusac_segmentation_evaluation import evaluate_segmentation_files
result = evaluate_segmentation_files("/share/lab_teng/trainee/tusharsingh/cell-seg/data/Monusac/rescaled/patched/all_merged/0002_train_0002_TCGA-73-4668_lung/00000x_00000y_mask.png",
                                     "/share/lab_teng/trainee/tusharsingh/cell-seg/inference/benchmarking/monusac/cellpose_sam/all_merged/0002_train_0002_TCGA-73-4668_lung/00000x_00000y_image.png", threshold=0.5)
print(result["instance_metrics"]["pq"])
print(result)