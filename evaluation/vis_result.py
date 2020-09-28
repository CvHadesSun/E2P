import torchvision.models as models

import torch

pthfile='tools/results/ape/inference/coco_linemod_ape_val/coco_results.pth'

coco_result=torch.load(pthfile)


# print(type(result))
# print(type(result))

# print(len(result))

# print(type(coco_result.results))

# print(coco_result.results.keys())

# print(type(coco_result.results["keypoints"]))

# print(coco_result.results["keypoints"].keys())

# print(type(coco_result.results["keypoints"]["dist_error"]))

#distance error

# print(type(coco_result.results["keypoints"]["dist_error"][0]))

# # print(coco_result.results["keypoints"]["dist_error"][0][(141,1)])

# dist_error=coco_result.results["keypoints"]["dist_error"][0]
# # print(len(dist_error))

# all_dist_error=0


# for e in dist_error:
#     all_dist_error+=dist_error[e][0]

# mean_error=all_dist_error/len(dist_error)

# print(mean_error)


print(coco_result.results["bbox"])


