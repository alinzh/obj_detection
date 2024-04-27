import fiftyone as fo
import fiftyone.zoo as foz

print(foz.list_zoo_datasets())

dataset = foz.load_zoo_dataset("coco-2017", split="validation")
dataset.name = "coco-2017-validation-example"

# Visualize the in the App
session = fo.launch_app(dataset)
