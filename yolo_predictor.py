import cv2
import numpy as np
import yaml
import os

CONFIDENCE, SCORE_THRESHOLD, IOU_THRESHOLD = 0.5, 0.5, 0.5
CONF_PATH = '/Users/alina/PycharmProjects/obj_detection/conf/main_conf.yaml'
font_scale, thickness = 1, 1


class Predictor():
    def __init__(self):
        with open(CONF_PATH, "r") as yamlfile:
            main_conf = yaml.load(yamlfile, Loader=yaml.FullLoader)

        self.yolo_conf, self.weights_path, self.processed_path = main_conf['yolo_conf'], main_conf['weights_path'], main_conf['processed_path']
        self.labels_path = main_conf['labels']
        self.labels = open(self.labels_path).read().strip().split("\n")
        self.path_data_dir = main_conf['dataset']
        self.dataset = os.listdir(main_conf['dataset'])
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.model = cv2.dnn.readNetFromDarknet(self.yolo_conf, self.weights_path)

    def get_img(self, img_name: str = None, cap=None) -> np.array:
        if cap:
            _, image = cap.read()
        else:
            image = cv2.imread(self.path_data_dir + '/' + img_name)
        h, w = image.shape[:2]
        transform_img = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

        return image, transform_img, h, w

    def make_prediction(self, input, h, w):
        self.model.setInput(input)

        # получаем имена всех слоев
        ln = self.model.getLayerNames()
        ln = [ln[i - 1] for i in self.model.getUnconnectedOutLayers()]
        # first 4 values represent the object's location:
        # - coordinates (x, y) for the center point,
        # - width and height of box
        layer_outputs = self.model.forward(ln)

        boxes, confidences, class_ids = [], [], []

        # for each layer in outputs
        for output in layer_outputs:
            # for each object in objects
            for detection in output:

                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # save objects just with height confidence
                if confidence > CONFIDENCE:

                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def draw_frame(self, boxes, confidences, class_ids, image):
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        if len(idxs) > 0:
            for i in idxs.flatten():
                # coordinates for rectangle
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # draw rectangle and text name
                color = [int(c) for c in self.colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                text = f"{self.labels[class_ids[i]]}: {confidences[i]:.2f}"

                (text_width, text_height) = \
                cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)

                # add opacity
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                # add text to image
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

        return image

    def val(self, samples=1, video_file=False):
        if not video_file:
            for i in range(samples):
                path = self.dataset[i]
                img, transform_img, h, w = self.get_img(path)
                boxes, confidences, class_ids = self.make_prediction(transform_img, h, w)
                res_img = self.draw_frame(boxes, confidences, class_ids, img)
                answer = cv2.imwrite(self.processed_path + f'result_example_{i}.jpg', res_img)
                if answer:
                    print('Image saved successfully')
                else:
                    print('Unable to save image')
        else:
            cap = cv2.VideoCapture(video_file)
            _, image = cap.read()
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter("/Users/alina/PycharmProjects/obj_detection/data/video/output.avi", fourcc, 20.0, (w, h))

            cnt = 0
            while True:
                cnt += 1
                img, transform_img, h, w = self.get_img(cap=cap)
                boxes, confidences, class_ids = self.make_prediction(transform_img, h, w)
                res_img = self.draw_frame(boxes, confidences, class_ids, img)
                out.write(res_img)
                out.release()
                cv2.destroyAllWindows()

                cv2.imshow("image", res_img)

                if ord("q") == cv2.waitKey(1):
                    break

            cap.release()
            cv2.destroyAllWindows()