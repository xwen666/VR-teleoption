from .arm import RobotArmController
from .yolo import YoloManager
from .realsense import RealsenseCamera
from .utils.log import get_logger_spdlog

logger = get_logger_spdlog()


def main():
    logger.info('== new instance')
    ctr = RobotArmController('192.168.1.18', 8080, 3)

    logger.info('setup cam and yolo ..')
    cam = RealsenseCamera()
    cam.get_info('color')
    cam.get_info('depth')
    cam.cosume_frames()
    cam.start_watching()

    yolo = YoloManager()
    yolo.load_models()
    yolo.set_detect_targets(['bottle'])

    logger.info('run the routine loop')
    ctr.run_main_routine(cam, yolo)

    # cam.print_sth()
    # yolo.debug_yolo(cam)
    # yolo.debug_sam2(cam, targets=['bottle', 'cup'])
    # yolo.debug_yolo_mini(cam)

    # cleanup
    logger.info('cleanup on exit')
    cam.stop_watching()
    ctr.close_modbus_mode(1)
    ctr.disconnect()


if __name__ == '__main__':
    main()
