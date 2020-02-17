'''
Motorcycle connecting rods inspection
'''


import plac

import cv2
import numpy as np
from scipy.spatial import distance as dist


BIT_QUADS = {
    '1': [
        np.array((
            [1, 0],
            [0, 0]),
            dtype="int"
        ),
        np.array((
            [0, 1],
            [0, 0]),
            dtype="int"
        ),
        np.array((
            [0, 0],
            [1, 0]),
            dtype="int"
        ),
        np.array((
            [0, 0],
            [0, 1]),
            dtype="int"
        )
    ],
    '3': [
        np.array((
            [0, 1],
            [1, 1]),
            dtype="int"
        ),
        np.array((
            [1, 0],
            [1, 1]),
            dtype="int"
        ),
        np.array((
            [1, 1],
            [0, 1]),
            dtype="int"
        ),
        np.array((
            [1, 1],
            [1, 0]),
            dtype="int"
        )
    ],
    'D': [
        np.array((
            [1, 0],
            [0, 1]),
            dtype="int"
        ),
        np.array((
            [0, 1],
            [1, 0]),
            dtype="int"
        )
    ]
}

AREA_THRESHOLD = 1200


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


def get_connected_component(labels, label):
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
    components_coords = [None] * num_labels
    for i in range(0, num_labels):
        components_coords[i] = np.fliplr(np.argwhere(labels == i))
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


def show_centroids(img, centroids, window_name, centroids_color=(255, 0, 0)):
    '''
    Show the given centroid on the image as small circles
    '''
    image = img.copy()
    for centroid in centroids:
        if centroid is not None:
            cv2.circle(
                image, (int(centroid[0]), int(centroid[1])),
                1, centroids_color, 1
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
            angles[i] = theta
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
        angles[i] = np.arctan(x_v1 / y_v1)
    return angles


def detach_rods(img):
    '''
    Detach rods in the given image, by applying
    morphological operations
    '''
    image = img.copy()
    kernel = np.ones((3, 3))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def holes_number(labels, num_labels):
    '''
    Compute the number of holes for each connected component,
    excluding the background
    '''
    n_holes = [None] * num_labels
    for i in range(1, num_labels):
        comp = get_connected_component(labels, i)
        holes = 1 - euler_number(comp, connectivity=8)
        n_holes[i] = holes
    return n_holes


def get_strides(img, window_size):
    '''
    Return a new matrix, which is the set of sliding windows
    on the original image, of the given size
    '''
    shape = (
        img.shape[0] - window_size + 1,
        img.shape[1] - window_size + 1,
        window_size, window_size
    )
    strides = 2 * img.strides
    patches = np.lib.stride_tricks.as_strided(
        img, shape=shape, strides=strides
    )
    patches = patches.reshape(-1, window_size, window_size)
    return patches


def euler_number(comp, connectivity=8):
    '''
    Compute the Euler number of the given image,
    containing a single connected component
    '''
    matches = {'1': 0, '3': 0, 'D': 0}
    patches = get_strides((comp.copy() / 255), window_size=2)
    for quad_type, kernels in BIT_QUADS.items():
        for kernel in kernels:
            for roi in patches:
                res = roi - kernel
                if cv2.countNonZero(res) == 0:
                    matches[quad_type] += 1
    euler = matches['1'] - matches['3']
    euler = (
        euler + 2 * matches['D'] if connectivity == 4
        else euler - 2 * matches['D']
    )
    return int(euler / 4)


def neighborhood(point, connectivity=8):
    '''
    Return the neighborhood of the given point
    '''
    x, y = tuple(point)
    return np.array([
        np.array([x - 1, y - 1]),
        np.array([x, y - 1]),
        np.array([x + 1, y - 1]),
        np.array([x - 1, y]),
        np.array([x + 1, y]),
        np.array([x - 1, y + 1]),
        np.array([x, y + 1]),
        np.array([x + 1, y + 1])
    ]) if connectivity == 8 else np.array([
        np.array([x, y - 1]),
        np.array([x - 1, y]),
        np.array([x + 1, y]),
        np.array([x, y + 1])
    ])


def find_contour(coords, connectivity=8):
    '''
    Find the contour of the given connected component
    '''
    contour = []
    for coord in coords:
        neighbors = neighborhood(coord, connectivity)
        for neighbor in neighbors:
            if not any(np.equal(coords, neighbor).all(1)):
                contour.append(coord)
                break
    return np.array(contour)


def haralick_circularity(contour, centroid):
    '''
    Compute Haralick's circularity measure
    '''
    distances = [dist.euclidean(coord, centroid) for coord in contour]
    std = np.std(distances)
    return np.mean(distances) / np.std(distances) if std > 0 else float('INF')


def find_circles(img, haralick_threshold=3):
    '''
    Find circles in the given image
    '''
    image = img.copy()
    num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=8
    )
    components_coords = get_components_coords(labels, num_labels)
    circles = []
    offset = 2
    for i, coords in enumerate(components_coords[offset:]):
        contour = find_contour(coords)
        circularity = haralick_circularity(
            contour, centroids[i + offset]
        )
        if circularity >= haralick_threshold:
            index = np.random.choice(contour.shape[0], 1, replace=False)
            radius = dist.euclidean(
                contour[index], centroids[i + offset]
            )
            circles.append((
                centroids[i + offset][0], centroids[i + offset][1], radius
            ))
    return circles


def show_circles(img, circles, window_name):
    '''
    Show the given circles on the given image,
    where a circle is (x_center, y_center, radius)
    '''
    image = img.copy()
    for (x, y, r) in circles:
        cv2.circle(image, (int(x), int(y)), 1, (0, 100, 100), 2)
        cv2.circle(image, (int(x), int(y)), int(r), (255, 0, 255), 2)
    show_image(image, window_name)


def get_blobs_mer(components_coords):
    '''
    Compute blobs minimum enclosing oriented rectangle
    '''
    blobs_mer = [None] * len(components_coords)
    for i, coords in enumerate(components_coords):
        blobs_mer[i] = np.int0(cv2.boxPoints(cv2.minAreaRect(coords)))
    return blobs_mer


def get_blobs_straight_bbox(components_coords):
    '''
    Compute blobs straight bounding rectangle
    '''
    blobs_bbox = [None] * len(components_coords)
    for i, coords in enumerate(components_coords):
        blobs_bbox[i] = cv2.boundingRect(coords)
    return blobs_bbox


def order_points(pts):
    '''
    Order the given vertices as
    (top-left, top-right, bottom-right, bottom-left)
    '''
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    (tr, br) = right_most
    return np.array([tl, tr, br, bl], dtype="float32")


def compute_mer_shape(mer):
    '''
    Compute lenght and width of the given oriented rectangle.
    Return (length, width)
    '''
    distances = [
        dist.euclidean(mer[0], mer[1]),
        dist.euclidean(mer[0], mer[2]),
        dist.euclidean(mer[0], mer[3])
    ]
    return (np.median(distances), np.min(distances))


def compute_blobs_shape(blobs_mer):
    '''
    Compute lenght and width of every blob
    '''
    blobs_shape = [None] * len(blobs_mer)
    for i, mer in enumerate(blobs_mer):
        blobs_shape[i] = compute_mer_shape(mer)
    return blobs_shape


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
            for shift in [0, np.pi / 2]:
                angle = angles[i] + shift
                point = (
                    centroid + img.shape[0] *
                    np.array([np.cos(angle), np.sin(angle)])
                )
                cv2.line(
                    image, (int(centroid[0]), int(centroid[1])),
                        (int(point[0]), int(point[1])), (0, 0, 255), 1
                )
    show_image(image, window_name)


def print_stats(n_holes, blobs_shape, centroids):
    '''
    Print some discovered properties
    '''
    for i in range(1, len(blobs_shape)):
        print(f'Connected component #{i}')
        print(f'- Centroid position: {centroids[i]}')
        print(f'- Rod type: {"A" if n_holes[i] == 1 else "B"}')
        print(f'- Length: {blobs_shape[i][0]}')
        print(f'- Width: {blobs_shape[i][1]}')
        if i != len(blobs_shape) - 1:
            print()


def filter_by_area(num_labels, labels, stats, centroids, area_threshold):
    '''
    Remove connected components which have a smaller area
    than the given one
    '''
    total_labels = num_labels
    to_remove = []
    for i in range(1, total_labels):
        area = stats[i][cv2.CC_STAT_AREA]
        if area < area_threshold:
            to_remove.append(i)
            labels[labels == i] = 0
            num_labels -= 1

    shift_labels = list(to_remove)
    new_labels = labels.copy()
    while shift_labels:
        i = shift_labels[0]
        if i >= total_labels:
            break
        shift_labels[0] = i + 1
        new_labels[new_labels == i + 1] = i
    return (
        num_labels,
        new_labels,
        np.delete(stats, to_remove, axis=0),
        np.delete(centroids, to_remove, axis=0)
    )


@plac.annotations(
    image_path=("Path to the image file", "option", "i", str)
)
def main(image_path='img/task-1/01.bmp'):
    '''
    Inspect the given connecting rod image
    '''
    # Load the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception('Please provide a valid image path.')

    # Apply Otsu threshold and invert the image
    _, threshed = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    inv_threshed = cv2.bitwise_not(threshed)
    show_image(inv_threshed, "Inverted threshed")

    # Detach rods
    # inv_threshed = detach_rods(inv_threshed)
    # show_image(inv_threshed, "Detached rods")

    # Compute connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inv_threshed, connectivity=8
    )
    show_connected_components(labels, window_name="Connected components")

    # Remove iron powder and distractors
    num_labels, labels, stats, centroids = filter_by_area(
        num_labels, labels, stats, centroids, AREA_THRESHOLD
    )
    show_connected_components(
        labels, window_name="Connected components after filtering"
    )

    # Compute connected components coordinates
    components_coords = get_components_coords(labels, num_labels)

    # Show circles
    circles = find_circles(threshed)
    show_circles(img, circles, window_name="Circles")

    # Compute blobs moments and show their orientation
    moments = compute_moments(components_coords, centroids)
    angles = get_blobs_orientation_from_moments(moments)
    show_blobs_axis(img, angles, centroids, "Major axis")

    # Show blobs oriented rectangles and compute shape features
    blobs_mer = get_blobs_mer(components_coords)
    show_blobs_mer(img, blobs_mer, "MER")
    blobs_shape = compute_blobs_shape(blobs_mer)

    # Compute the number of holes in each component
    n_holes = holes_number(labels, num_labels)

    # Print connected components analysis
    print_stats(n_holes, blobs_shape, centroids)


if __name__ == '__main__':
    plac.call(main)
