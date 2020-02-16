from pathlib import Path

import cv2
import numpy as np
import skimage


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


def show_image(image, window_name, wait=True):
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


def get_connected_component(labels, label, img_size):
    '''
    Return a specific connected component on a new image
    '''
    mask = np.zeros_like(labels, dtype=np.uint8)
    mask[labels == label] = 255
    return mask


def get_components_coords(labels, num_labels):
    '''
    Compute connected components coordinates
    '''
    components_coords = []
    for i in range(0, num_labels):
        components_coords.append(np.fliplr(np.argwhere(labels == i)))
    return components_coords


def moment(coords, m, n):
    '''
    Compute moment of order (m, n) with the given coordinates
    '''
    def compute_moment(v):
        return (v[0] ** m) * (v[1] ** n)

    return np.sum(np.apply_along_axis(compute_moment, axis=1, arr=coords))


def central_moment(coords, centroid, m, n):
    '''
    Compute central moment of order (m, n) with the given coordinates
    '''
    xc, yc = centroid[0], centroid[1]

    def compute_central_moment(v):
        return ((v[0] - xc) ** m) * ((v[1] - yc) ** n)

    return np.sum(np.apply_along_axis(compute_central_moment, axis=1, arr=coords))


def compute_moments(components_coords, centroids):
    '''
    Compute connected components moments
    '''
    product = list((m, n) for m in range(3) for n in range(3))
    num_labels = len(centroids)
    moments = [None] * num_labels
    for i in range(1, num_labels):
        moments[i] = dict()
        for m, n in product:
            moments[i][f'm{m}{n}'] = moment(
                components_coords[i], m, n
            )
            moments[i][f'mu{m}{n}'] = central_moment(
                components_coords[i], centroids[i], m, n
            )
    return moments


def compute_centroids(components_coords):
    '''
    Compute blobs centroids from coordinates
    '''
    centroids = [None] * len(components_coords)
    for i, coords in enumerate(components_coords):
        centroids[i] = np.sum(coords, axis=0) / len(coords)
    return centroids


def compute_centroids_from_moments(moments):
    '''
    Compute blobs centroids from moments
    '''
    centroids = [None] * len(moments)
    for i, blob_moments in enumerate(moments):
        if blob_moments is not None:
            centroid_x = int(blob_moments["m10"] / blob_moments["m00"])
            centroid_y = int(blob_moments["m01"] / blob_moments["m00"])
            centroids[i] = (centroid_x, centroid_y)
    return centroids


def show_centroids(img, centroids, window_name):
    '''
    Show the given centroid on the image as small circles
    '''
    image = img.copy()
    for centroid in centroids:
        if centroid is not None:
            cv2.circle(
                image, (int(centroid[0]), int(centroid[1])), 1, (255, 0, 0), 1
            )
    show_image(image, window_name)


def get_blobs_orientation_from_moments(moments):
    '''
    Compute blobs orientation from central moments
    '''
    angles = [None] * len(moments)
    for i, blob_moments in enumerate(moments):
        if blob_moments is not None:
            theta = -0.5 * np.arctan(
                (2 * blob_moments['mu11']) /
                (blob_moments['mu02'] - blob_moments['mu20'] + 1e-5)
            )
            print(theta)
            angles[i] = {
                'major': theta,
                'minor': theta + (np.pi / 2)
            }
    return angles


def get_blobs_orientation_from_cov(components_coords, centroids):
    '''
    Compute blobs orientation from the covariance matrix
    associated with the components coordinates
    '''
    angles = [None] * len(components_coords)
    for i, coords in enumerate(components_coords):
        xy = np.transpose(coords)
        x = xy[0] - centroids[i][0]
        y = xy[1] - centroids[i][1]
        values = np.vstack([x, y])
        cov = np.cov(values)
        evals, evecs = np.linalg.eig(cov)
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        angles[i] = {
            'major': np.arctan(x_v1 / y_v1),
            'minor': np.arctan(x_v2 / y_v2)
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


def show_blobs_axis(img, angles, centroids, window_name):
    '''
    Show blobs major and minor axis as a line
    through the barycentre
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
    show_image(image, window_name)


def main():
    img = cv2.imread('img/task-1/04.bmp', cv2.IMREAD_GRAYSCALE)
    _, threshed = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    inv_threshed = cv2.bitwise_not(threshed)
    show_image(inv_threshed, "Inverted threshed")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inv_threshed, connectivity=8
    )
    show_connected_components(labels, window_name="Connected components")
    components_coords = get_components_coords(labels, num_labels)
    moments = compute_moments(components_coords, centroids)
    show_centroids(img, centroids, "Centroids")
    angles = get_blobs_orientation_from_moments(moments)
    angles = get_blobs_orientation_from_cov(components_coords, centroids)
    show_blobs_axis(img, angles, centroids, "Major axis")
    blobs_mer = get_blobs_mer(components_coords)
    show_blobs_mer(img, blobs_mer, "MER")
    # blobs_bbox = get_blobs_straight_bbox(components_coords)
    # show_blobs_straight_bbox(threshed, blobs_bbox, "BBOX")


if __name__ == '__main__':
    main()
