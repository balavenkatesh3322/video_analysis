import cv2
import argparse
import numpy as np
import os


class_file = "yolov3.txt"
weight_file = "yolov3.weights"
config_file = "yolov3.cfg"

#argument parser to read input from comment line
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=False,
                help = 'path to input video')
args = ap.parse_args()


if args.video is None:
    video_file_name = "videos/video_0.7177935927284033.mp4"
else:
    video_file_name = args.video
 
#to get file name    
file_name, file_extension = os.path.splitext(video_file_name)

#detected object names saved in set
object_name = set()


def get_output_layers(net):
    """To get output layer.
    Args:
        net
    Returns:
        output_layers.
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """Draw prediction layer on box.
    Args:
        img:
        class_id
        confidence
        x,
        y, 
        x_plus_w, 
        y_plus_h
    Returns:
        None
    """
    label = str(classes[class_id])
    object_name.add(label)
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def create_folder_for_image():
    """To store extracted image from video file
    Args:
        video_file_name:
    Returns:
        file_name
    """
    if not os.path.exists(file_name): 
        os.makedirs(file_name)
    return file_name


def save_object_names():
    """Save object name in json file
    """
    with open(file_name + ".json", "w") as f:
        f.write(str(object_name))


#Read video files
cam = cv2.VideoCapture(video_file_name)


# frame 
currentframe = 0
  
while(True): 
      
    # reading from frame 
    ret,image = cam.read() 
  
    if ret: 
        file_name = create_folder_for_image() 
        name = file_name  + "/" + str(currentframe) + '.jpg'
        print ('Creating...' + name) 
  
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        #Reading class names from class file
        with open(class_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(weight_file, config_file)

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        #non-max suppression to ignore weak bounding box
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        # writing the extracted images
        cv2.imwrite(name, image) 
        cv2.waitKey()  
        cv2.destroyAllWindows()

        currentframe += 1
    else:
        save_object_names()
        break



