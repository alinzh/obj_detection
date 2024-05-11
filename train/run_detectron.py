from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
import os, json, cv2, random
from IPython import display
import PIL
from detectron2.engine import DefaultTrainer
import pickle
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
    a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
      (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
      image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(PIL.Image.fromarray(a))


def register_dataset():
    register_coco_instances(
        f"traffic_sign_train", {},
        f"/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset/train/train_anno.json",
    "/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset/train/images"
    )
    register_coco_instances(
        f"traffic_sign_test", {},
        f"/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset/valid/val_anno.json",
    "/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset/valid/images"
    )

    # visualize training data
    my_dataset_train_metadata = MetadataCatalog.get("traffic_sign_train")
    dataset_dicts = DatasetCatalog.get("traffic_sign_train")

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2_imshow(vis.get_image()[:, :, ::-1])


def train():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("traffic_sign_train",)
    cfg.DATASETS.TEST = ("traffic_sign_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 156

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    with open("/Users/alina/PycharmProjects/obj_detection/models/detectron2/cfg.pkl", "wb") as f:
        pickle.dump(cfg, f)

    return cfg, trainer


def show_predicts(cfg):
    predictor = DefaultPredictor(cfg)

    my_dataset_test_metadata = MetadataCatalog.get("traffic_sign_test")
    dataset_dicts = DatasetCatalog.get("traffic_sign_test")

    for d in random.sample(dataset_dicts, 5):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=my_dataset_test_metadata,
                       scale=0.5
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(out.get_image()[:, :, ::-1])


def evaluate(cfg, trainer):
    evaluator = COCOEvaluator("traffic_sign_val", cfg, False, output_dir="/Users/alina/PycharmProjects/obj_detection/train/output")
    val_loader = build_detection_test_loader(cfg, "traffic_sign_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)
    return cfg, trainer


def load_config(config_path):
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    return cfg


def predict(cfg, model_weights, image_folder, output_folder):
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    image_files = os.listdir(image_folder)
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        original_image = cv2.imread(image_path)
        outputs = predictor(original_image)

        output_path = os.path.join(output_folder, image_file.replace(".jpg", "_prediction.jpg"))
        v = Visualizer(original_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2, instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(output_path, v.get_image()[:, :, ::-1])


if __name__ == '__main__':
    config_path = "/Users/alina/PycharmProjects/obj_detection/models/detectron2/cfg.pkl"
    register_dataset()
    cfg, trainer = train()

    image_folder = "/Users/alina/PycharmProjects/obj_detection/datasets/traffic_sign/dataset/valid/images/"
    output_folder = "/Users/alina/PycharmProjects/obj_detection/data/from_detectron2/"
    model_weights = "/Users/alina/PycharmProjects/obj_detection/train/output/model_final.pth"

    cfg = load_config(config_path)
    predict(cfg, model_weights, image_folder, output_folder)

