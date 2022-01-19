import os.path

from PyQt5.QtCore import QDir


class UiShare:
    """
        Ui
    """
    # ui
    mainWin = None
    loginWin = None

    mode = None  # detect or track
    ways = None  # 1:online 0:offline

    algorithm_detect = None
    algorithm_track = None


# =====================================================================================
class DetectShare:
    """
        Detect attribute
    """
    # detect model
    list_media = []  # import images or video list
    list_cfgs = []  # model config files list
    list_weights = []  # model weights files list
    list_names = []  # label files list

    detector = None  # Detect Model(class)
    detector_algorithm = None  # str

    save_path = os.path.join(QDir.currentPath(), 'tmp')
    save_detect_result = False  # bool
    result = None


# =====================================================================================
class TrackShare:
    """
        Track attribute
    """
    list_media = []  # import images or video list
    # detect
    detect_list_cfgs = []  # model config files list
    detect_list_weights = []  # model weights files list
    detect_list_names = []  # label files list

    detector = None
    detector_algorithm = None

    # track
    tracker = None
    tracker_algorithm = None

    # appearance
    appearance = None
    appearance_list_weights = []

    # else
    vis = None
    seq_info = None
    metric = None
    frame_idx = 0
    results = []

    save_video = True  # When test the function of UI False, other True
    save_track_result = False  # bool
    save_path = os.path.join(QDir.currentPath(), 'tmp')

    writer_video = None

    result_video = None  # video tmp path


class Video:
    VIDEO_TYPE_OFFLINE = 0  # 0: offline  1: realTime
    VIDEO_TYPE_REAL_TIME = 1

    STATUS_INIT = 0  # 0: init 1:playing 2: pause
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    video_url = ""

    fps = 0

    auto_play = False

    status = 0

    playCapture = None

    video_type = 0  # 0: offline  1: realTime

    current_frame = None


class Analysis:
    label_path = ''  # label source
    output_path = ''  # model predict output


