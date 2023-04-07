import os
import glob
import numpy as np
import random
import cv2
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path

# -----
# Dataset Generation Parameters
# -----
pdf_data_path = "dataset/genuine docs/PDFs"
genuine_img_data_path = "dataset/genuine docs/images"
forged_img_data_root = "dataset/forged docs"

min_shift_px_x = 1
min_shift_px_y = 1
max_shift_px_x = 5
max_shift_px_y = 5

min_scale_px_x = 1.1
min_scale_px_y = 1.1
max_scale_px_x = 1.2
max_scale_px_y = 1.2

shift_prob = 0.1
scale_prob = 0.1

# -----
# PDF to img conversion
# -----
def pdf_to_img(pdf_data_path, genuine_img_data_path):
    i = 0
    for pdf_path in glob.iglob(os.path.join(pdf_data_path, "*.pdf")):
        pages = convert_from_path(pdf_path)
        for page in pages:
            page.save(os.path.join(genuine_img_data_path, f"{i}.jpeg"), "JPEG")
            i += 1

pdf_to_img(pdf_data_path, genuine_img_data_path)


# -----
# Forgeries creation
# -----
possible_shift_values_x = [px for px in range(min_shift_px_x, max_shift_px_x)] + [-x for x in range(min_shift_px_x, max_shift_px_x)]
possible_shift_values_y = [px for px in range(min_shift_px_y, max_shift_px_y)] + [-x for x in range(min_shift_px_y, max_shift_px_y)]
possible_scale_values_x = [px for px in np.arange(min_scale_px_x, max_scale_px_x, 0.5)]# + [-px for px in np.arange(1.0, max_shift_px_x, 0.5)]
possible_scale_values_y = [py for py in np.arange(min_scale_px_y, max_scale_px_y, 0.5)]# + [-py for py in np.arange(1.0, max_shift_px_y, 0.5)]

for path in glob.iglob(os.path.join(genuine_img_data_path, '*.jpeg')):
    # loading the image
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # giving the same value to bgr channels
    backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # so that its possible to only keep 1 channel
    img = backtorgb
    img_height, img_width, _ = img.shape
    alt_img = img.copy()
    alt_img_gt = img.copy()

    # detecting characters in image
    d = pytesseract.image_to_boxes(img, output_type=Output.DICT)
    n_boxes = len(d['char'])
    
    for i in range(n_boxes):
        # ocr output for char_i
        text, x1, y2, x2, y1 = d['char'][i], d['left'][i], d['top'][i], d['right'][i], d['bottom'][i]
        
        # skipping boxes that are too small
        if x2-x1<10 or y2-y1<10:
            continue

        # coloring altered letters to label them
        patch = img[img_height-y2:img_height-y1, x1:x2]
        col_patch = patch.copy()
        col_patch[:, :, 0] = 0
        col_patch[:, :, 2] = 0
        
        # randomly altering characters
        r = random.random()
        if r<shift_prob:
            # shifting characters
            x_shift = random.choice(possible_shift_values_x)
            y_shift = random.choice(possible_shift_values_y)

            alt_img[img_height-y2:img_height-y1, x1:x2] = 255
            alt_img_gt[img_height-y2:img_height-y1, x1:x2] = 255
            try:
                alt_img[img_height-y2+y_shift : img_height-y1+y_shift , x1+x_shift : x2+x_shift] = patch
                alt_img_gt[img_height-y2+y_shift : img_height-y1+y_shift , x1+x_shift : x2+x_shift] = col_patch
            except: 
                alt_img[img_height-y2 : img_height-y1 , x1 : x2] = patch # undo in case patch x,y are out of bounds
                alt_img_gt[img_height-y2 : img_height-y1 , x1 : x2] = patch # undo
        
        elif r<shift_prob + scale_prob:
            # scaling characters
            x_scale = random.choice(possible_scale_values_x)
            y_scale = random.choice(possible_scale_values_y)
            
            width = x2-x1
            height = y2-y1
            scaled_w = int(x_scale*width)
            scaled_h = int(y_scale*height)

            sc_patch = cv2.resize(patch, (scaled_w, scaled_h))
            sc_patch_col = cv2.resize(col_patch, (scaled_w, scaled_h))
            
            alt_img[img_height-y2:img_height-y1, x1:x2] = 255
            alt_img_gt[img_height-y2:img_height-y1, x1:x2] = 255

            sc_x_shift = scaled_w-width
            sc_y_shift = scaled_h-height
            px = sc_x_shift//2 != sc_x_shift/2
            py = sc_y_shift//2 != sc_y_shift/2
            try:
                alt_img[img_height-y2-sc_y_shift//2 : img_height-y1+sc_y_shift//2+px , x1-sc_x_shift//2 : x2+sc_x_shift//2+py] = sc_patch
                alt_img_gt[img_height-y2-sc_y_shift//2 : img_height-y1+sc_y_shift//2+px , x1-sc_x_shift//2 : x2+sc_x_shift//2+py] = sc_patch_col
            except: 
                alt_img[img_height-y2-sc_y_shift//2 : img_height-y1+sc_y_shift//2+px , x1-sc_x_shift//2 : x2+sc_x_shift//2+py] = 255
                alt_img[img_height-y2:img_height-y1, x1:x2] = patch # undo
                alt_img_gt[img_height-y2-sc_y_shift//2 : img_height-y1+sc_y_shift//2+px , x1-sc_x_shift//2 : x2+sc_x_shift//2+py] = 255
                alt_img_gt[img_height-y2:img_height-y1, x1:x2] = patch # undo

    cv2.imwrite(os.path.join(forged_img_data_root, "forgeries", path.split(os.sep)[-1]), alt_img)
    cv2.imwrite(os.path.join(forged_img_data_root, "ground truth", path.split(os.sep)[-1]), alt_img_gt)
