def normalize_images(images):
    return images / 255.0


def sliding_window(image, step_size, window_size):
    """Slide a window across the image."""
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y : y + window_size[1], x : x + window_size[0]])
