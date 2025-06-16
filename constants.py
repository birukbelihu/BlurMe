def get_app_name() -> str:
    return "Blur Me"

def get_prototxt_file() -> str:
    return "models/deploy.prototxt"


def get_caffe_model() -> str:
    return "models/face_detector.caffemodel"


def exit_keys() -> tuple:
    return 'q', 'Q', 'e', 'E'
