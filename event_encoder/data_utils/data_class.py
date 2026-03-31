
""" DSEC 11 classes """
DSEC11_CLASS = [
    'background', 
    'building', 
    'fence', 
    'person', 
    'pole', 
    'road',
    'sidewalk', 
    'vegetation', 
    'car', 
    'wall', 
    'traffic sign'
]

DSEC11_INSTANCE_ID = [1, 2, 3, 4, 7, 8, 10]

DSEC11_PRED_TO_ID = {
    0:1, 1:2, 2:3, 3:4, 4:7, 5:8, 6:10
}


""" DSEC 19 classes """
DSEC19_CLASS = [
    "road", 
    "sidewalk", 
    "building",
    "wall", 
    "fence", 
    "pole", 
    "traffic light", 
    "traffic sign",
    "vegetation", 
    "terrain", 
    "sky", 
    "person", 
    "rider", 
    "car", 
    "truck", 
    "bus", 
    "train", 
    "motorcycle",
    "bicycle"
]

DSEC19_INSTANCE_ID = [2, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]

DSEC19_PRED_TO_ID = {
    0:2, 1:4, 2:5, 3:6, 4:7, 5:8, 6:11, 7:12, 8:13, 9:14, 10:15, 11:16, 12:17, 13:18 
}


""" DDD17 classes """
DDD17_CLASS = [
    "flat",
    "construction+sky",
    "object",
    "nature",
    "human",
    "vehicle"
]

DDD17_INSTANCE_ID = [1, 2, 3, 4, 5]

DDD17_PRED_TO_ID = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5
}


""" DSEC PART classes """
DSEC_PART_CLASS = [
    "vehicle", 
    "vehicle: wheel", 
    "vehicle: window", 
    "vehicle: light", 
    "vehicle: side mirror", 
    "vehicle: license plate", 
    "vehicle: door", 
    "building", 
    "building: window", 
    "building: roof", 
    "building: door", 
    "building: terrace" 
]

DSEC_PART_INSTANCE_ID = [1, 2, 3, 4, 5, 8, 9, 10, 11]

DSEC_PART_PRED_TO_ID = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 8, 6: 9, 7: 10, 8: 11
}
