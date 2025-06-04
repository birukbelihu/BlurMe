def prototxt_file() -> str:
    return "models/deploy.prototxt"


def caffe_model() -> str:
    return "models/face_detector.caffemodel"


def exit_keys() -> tuple:
    return 'q', 'Q', 'e', 'E'
