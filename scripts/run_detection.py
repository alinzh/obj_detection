from yolo_predictor import Predictor


def run_on_images(yolo: Predictor, samples: int) -> None:
    yolo.val(samples=samples)


def run_on_video(yolo: Predictor, video_path: str):
    yolo.val(video_file=video_path)


if __name__ == "__main__":
    yolo = Predictor()
    run_on_video(
        yolo,
        "/Users/alina/PycharmProjects/obj_detection/data/video/video_test_short.mp4",
    )
