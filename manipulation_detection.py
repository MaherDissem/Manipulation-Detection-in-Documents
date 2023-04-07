# -----
# OCR Graph Features for Manipulation Detection in Documents
# paper implementation
# -----
import os 
import glob
import numpy as np
import pandas as pd
import cv2
import pytesseract
from pytesseract import Output
from collections import deque


class BoundingBox():

    def __init__(self, text='', x1=0, y2=0, x2=0, y1=0, patch=None):
        self.text = text
        self.x1 = x1
        self.y2 = y2
        self.x2 = x2
        self.y1 = y1
        self.patch = patch
        self.height = y2-y1
        self.width = x2-x1
    
    def get_hu_moments(self):
        if self.patch is None:
            return np.array([0 for _ in range(7)])
        return cv2.HuMoments(cv2.moments(self.patch)).flatten() # list

    def get_distance_to_node(self, node):
        x_mid = (self.x1 + self.x2)/2
        y_mid = (self.y1 + self.y2)/2
        x_mid_c_node = (node.x1 + node.x2)/2
        y_mid_c_node = (node.y1 + node.y2)/2
        euclidian_dist = np.sqrt((x_mid-x_mid_c_node)**2 + (y_mid-y_mid_c_node)**2)/2
        y_val_dist = y_mid - y_mid_c_node
        return euclidian_dist, y_val_dist


def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ , thresh = cv2.threshold(gray, 150, 255, 0)
    return thresh


def label_and_split_by_line(ocr_out, img, lab_img=None, alpha=1.35):
    n_boxes = len(ocr_out['char'])
    img_height, _ = img.shape
    img_with_boxes = img.copy()
    y_labels = []
    same_line = []
    lines = []
    i_line = -1
    deque_size = 1
    prev_y1_deque = deque([ocr_out['bottom'][0]], maxlen=deque_size)
    prev_h_deque = deque([ocr_out['top'][0]-ocr_out['bottom'][0]], maxlen=deque_size)

    for i in range(n_boxes):
        # ocr output for char_i
        text, x1, y2, x2, y1 = ocr_out['char'][i], ocr_out['left'][i], ocr_out['top'][i], ocr_out['right'][i], ocr_out['bottom'][i]
        # skipping boxes that are too small
        if x2-x1<10 or y2-y1<10:
            continue     
        
        patch = img[img_height-y2:img_height-y1, x1:x2]
        box = BoundingBox(text, x1, y2, x2, y1, patch)
        height = box.height
        # detecting character's label
        if lab_img is not None: # None during inference
            lab_patch = lab_img[img_height-y2:img_height-y1, x1:x2]
            hsv = cv2.cvtColor(lab_patch, cv2.COLOR_BGR2HSV)
            lower_val = np.array([37,42,0]) 
            upper_val = np.array([84,255,255]) 
            mask = cv2.inRange(hsv, lower_val, upper_val)
            hasGreen = np.sum(mask)/255
            y_labels.append(hasGreen>box.height*box.width/1.5)
        # detect new lines
        avg_prev_y1 = np.mean(prev_y1_deque)
        avg_prev_h = np.mean(prev_h_deque)
        # new line
        if np.abs(avg_prev_y1-y1) > alpha*max(height, avg_prev_h):
            i_line += 1
            if same_line!=[]:
                lines.append(same_line)
            same_line = [box]
        # same line
        else:
            same_line.append(box)
        prev_y1_deque.append(y1)
        prev_h_deque.append(height)
        # draw ocr-detected characters
        cv2.rectangle(img_with_boxes, (x1, img_height-y1), (x2, img_height-y2) , (0,255,0), 2)
    cv2.imwrite('output_images/detected_chars.png', img_with_boxes)
    lines.append(same_line)

    return lines, y_labels


def build_graphs(lines, n_neighbors):
    n_nodes = 2*n_neighbors+1
    graphs = []
    for same_line in lines:
        if len(same_line)<n_nodes:
            # empty nodes as pad
            l_pad = [BoundingBox() for _ in range(n_neighbors)]
            r_pad = l_pad
        else:
            # padding with nodes from other side of the line
            l_pad = same_line[-n_neighbors:]
            r_pad = same_line[:n_neighbors]
        # padding graph nodes
        padded_line = l_pad
        padded_line.extend(same_line)
        padded_line.extend(r_pad)

        j = 0
        while j<len(same_line):
            kernel = padded_line[j: n_nodes+j]
            graph = np.array([])
            for node in kernel:
                # building features
                height = node.height
                width = node.width
                central_node = kernel[n_neighbors]
                euclidian_dist, y_val_dist = node.get_distance_to_node(central_node)
                hu_moments = node.get_hu_moments()
                # saving features
                node_feat = np.concatenate([np.array([height, width, y_val_dist, euclidian_dist]),hu_moments])
                graph = np.concatenate([graph, node_feat])
            graphs.append(graph)
            j += 1     
    return graphs


def get_features(dataset_path, n_neighbors, alpha):
    X, y = [], []
    for path in glob.iglob(os.path.join(dataset_path, "forgeries", "*.jpeg")):
        # imgs path
        img_name = path.split(os.sep)[-1]
        img_path = os.path.join(dataset_path, "forgeries", img_name)
        gt_img_path = os.path.join(dataset_path, "ground truth", img_name)
        # loading images
        img = cv2.imread(img_path)
        lab_img = cv2.imread(gt_img_path)
        # detecting characters in image
        img = preprocess_img(img)
        ocr_out = pytesseract.image_to_boxes(img, output_type=Output.DICT)
        # split detected text by line and label characters
        lines, y_labels = label_and_split_by_line(ocr_out, img, lab_img, alpha=alpha)
        # build character level features
        graphs = build_graphs(lines, n_neighbors)
        # append single img results
        X += graphs
        y += y_labels
    # returns X, y features
    return np.array(X), np.array(y).reshape(-1,)


def infer_img(img_path, n_neighbors, alpha):
    img = cv2.imread(img_path)
    pprc_img = preprocess_img(img)
    img_height = img.shape[0]

    ocr_out = pytesseract.image_to_boxes(pprc_img, output_type=Output.DICT)
    lines, _ = label_and_split_by_line(ocr_out, pprc_img, lab_img=None, alpha=alpha)
    graphs = build_graphs(lines, n_neighbors)
    X = np.array(graphs)
    model = xgb.XGBClassifier()
    model.load_model("weights/xgb")
    y_pred = model.predict(X)

    # drawing detected forgeries
    detected_forgeries = img.copy()
    boxes = []
    for i in range(len(lines)):
        boxes.extend(lines[i])
    for i, box in enumerate(boxes):
        if y_pred[i]==True:
            cv2.rectangle(detected_forgeries, (box.x1, img_height-box.y1), (box.x2, img_height-box.y2) , (0,255,0), 2)
    cv2.imwrite('output_images/detected_forgeries.png', detected_forgeries)

    return y_pred


# -----
# Supervised Classification
# -----
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset_path = "dataset/forged docs"
alpha = 1.35 # alpha should be adjusted according to the document's line spacing
n_neighbors = 5 

X, y = get_features(dataset_path, n_neighbors, alpha)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # switch to testing on a whole img instead of splitting graphs ?

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)
model.save_model("weights/xgb")

# infer_img("dataset/forged docs/forgeries/11.jpeg", n_neighbors, alpha)
