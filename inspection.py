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


def get_components_coords(labels):
    '''
    Compute connected components coordinates
    '''
    components_coords = []
    num_labels = np.max(labels) + 1
    for i in range(num_labels):
        components_coords.append(np.argwhere(labels == i))
    return components_coords


def get_moments(components_coords):
    '''
    Compute connected components moments
    '''
    num_labels = len(components_coords)
    moments = [None] * num_labels
    for i in range(1, num_labels):
        moments[i] = cv2.moments(components_coords[i])
    return moments


def get_blobs_orientation(moments):
    '''
    Compute blobs orientation from central moments
    '''
    angles = [None] * len(moments)
    for i, blob_moments in enumerate(moments):
        if blob_moments is not None:
            major_axis_angle = -0.5 * np.arctan(
                blob_moments['mu11'] /
                (blob_moments['mu02'] + blob_moments['mu20'] + 1e-5)
            )
            minor_axis_angle = major_axis_angle + (np.pi / 2)
            angles[i] = {
                'major': major_axis_angle,
                'minor': minor_axis_angle
            }
    return angles


def get_blobs_mer(components_coords):
    '''
    Compute blobs minimum enclosing oriented rectangle
    '''
    blobs_mer = [None]
    for _, coords in enumerate(components_coords):
        blobs_mer.append(
            np.int0(cv2.boxPoints(cv2.minAreaRect(coords)))
        )
    return blobs_mer


def get_blobs_straight_bbox(components_coords):
    '''
    Compute blobs straight bounding rectangle
    '''
    blobs_bbox = [None]
    for _, coords in enumerate(components_coords):
        blobs_bbox.append(cv2.boundingRect(coords))
    return blobs_bbox


def show_blobs_mer(img, blobs_mer, window_name):
    '''
    Show the given oriented rectangles on the given image
    '''
    image = img.copy()
    for mer in blobs_mer:
        if mer is not None:
            cv2.drawContours(image, [mer], 0, (0, 0, 255), 1)
    show_image(image, window_name)


def show_blobs_straight_bbox(img, blobs_bbox, window_name):
    '''
    Show the given straight rectangles on the given image
    '''
    image = img.copy()
    for bbox in blobs_bbox:
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    show_image(image, window_name)


def show_blobs_axis(img, angles, centroids):
    '''
    '''
    image = img.copy()
    for i, centroid in enumerate(centroids):
        if angles[i] is not None:
            major_axis_angle = angles[i]['major']
            point = (
                centroid + img.shape[0] *
                np.array([np.cos(major_axis_angle), np.sin(major_axis_angle)])
            )
            cv2.line(
                image, (int(centroid[0]), int(centroid[1])),
                (int(point[0]), int(point[1])), (0, 0, 255), 1
            )
    show_image(image, "TEST")


def main():
    img = cv2.imread('img/task-1/01.bmp', cv2.IMREAD_GRAYSCALE)
    _, threshed = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    show_image(threshed, "Threshed")
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(threshed)
    print(centroids)
    show_connected_components(labels, "Connected components")
    components_coords = get_components_coords(labels)
    print(components_coords)
    moments = get_moments(components_coords)
    angles = get_blobs_orientation(moments)
    # blobs_mer = get_blobs_mer(components_coords)
    # show_blobs_mer(threshed, blobs_mer, "MER")
    # blobs_bbox=get_blobs_straight_bbox(components_coords)
    # show_blobs_straight_bbox(threshed, blobs_bbox, "BBOX")
    show_blobs_axis(threshed, angles, centroids)


if __name__ == '__main__':
    main()
