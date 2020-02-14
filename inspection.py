from pathlib import Path

import cv2
import numpy as np


IMG_DIR = 'img'
IMAGE_PATHS = {
    'task-1': [],
    'task-2': {
        'change-1': [],
        'change-2': [],
        'change-3': [],
    }
}
IMAGES = {
    'task-1': [],
    'task-2': {
        'change-1': [],
        'change-2': [],
        'change-3': [],
    }
}


def read_images():
    IMAGE_PATHS['task-1'] = [
        path.name for path in Path(f'{IMG_DIR}/task-1').rglob('*.bmp')
    ]
    for i in range(1, 4):
        IMAGE_PATHS['task-2'][f'change-{i}'] = [
            path.name for path in Path(f'{IMG_DIR}/task-2/change-{i}').rglob('*.bmp')
        ]


def show_image(image: np.ndarray, window_name: str, wait: bool = True) -> None:
    '''
    Show an image and eventually wait until the user press a key
    '''
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    if wait:
        cv2.waitKey()


def show_connected_components(labels, window_name):
	'''
	Show connected components using pseudo-colors
	'''
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    show_image(labeled_img, window_name)


def get_contours(img):
    '''
    Find the contours of the given connected components
    '''
	pass
    


def main():
    img = cv2.imread('img/task-1/01.bmp', cv2.IMREAD_GRAYSCALE)
    _, threshed = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    # show_image(threshed, "Threshed")
    ret, labels = cv2.connectedComponents(threshed)
    # show_connected_components(labels, "Connected components")
    get_holes(threshed)


if __name__ == '__main__':
    main()
