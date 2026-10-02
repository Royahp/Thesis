from PIL import Image

from src.metrics.spatial_metrics import parse_bounds


def crop_widget(image_path, bounds):
    # Crop a widget from the screenshot
    parsed_bounds = parse_bounds(bounds)

    if parsed_bounds is None:
        return None

    x1, y1, x2, y2 = parsed_bounds

    image = Image.open(image_path).convert("RGB")

    return image.crop((
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    ))