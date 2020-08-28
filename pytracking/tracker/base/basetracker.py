import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time
import numpy as np


class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params

    def initialize(self, image, state, class_info=None):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        self.initialize(image, sequence.init_state)
        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)

        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, sequence.init_state)

        # Track
        tracked_bb = [sequence.init_state]
        for frame in sequence.frames[1:]:
            image = self._read_image(frame)

            start_time = time.time()
            state = self.track(image)
            times.append(time.time() - start_time)

            tracked_bb.append(state)

            if self.params.visualization:
                self.visualize(image, state)

        return tracked_bb, times

    def _to_polygon(self, polys):
        from shapely.geometry import Polygon, box
        def to_polygon(x):
            assert len(x) in [4, 8]
            if len(x) == 4:
                return box(x[0], x[1], x[0] + x[2], x[1] + x[3])
            elif len(x) == 8:
                return Polygon([(x[2 * i], x[2 * i + 1]) for i in range(4)])

        if polys.ndim == 1:
            return to_polygon(polys)
        else:
            return [to_polygon(t) for t in polys]

    def poly_iou(self, polys1, polys2, bound=None):
        assert polys1.ndim in [1, 2]
        if polys1.ndim == 1:
            polys1 = np.array([polys1])
            polys2 = np.array([polys2])
        assert len(polys1) == len(polys2)

        polys1 = self._to_polygon(polys1)
        polys2 = self._to_polygon(polys2)
        if bound is not None:
            bound = box(0, 0, bound[0], bound[1])
            polys1 = [p.intersection(bound) for p in polys1]
            polys2 = [p.intersection(bound) for p in polys2]

        eps = np.finfo(float).eps
        ious = []
        for poly1, poly2 in zip(polys1, polys2):
            area_inter = poly1.intersection(poly2).area
            area_union = poly1.union(poly2).area
            ious.append(area_inter / (area_union + eps))
        ious = np.clip(ious, 0.0, 1.0)

        return ious

    def track_sequence_vot(self, sequence):
        """Run tracker on a vot sequence."""

        if len(sequence.ground_truth_rect[0]) == 4:
        	x = sequence.ground_truth_rect[:, [0]]
        	y = sequence.ground_truth_rect[:, [1]]
        	w = sequence.ground_truth_rect[:, [2]]
        	h = sequence.ground_truth_rect[:, [3]]
        	sequence.ground_truth_rect = np.concatenate((x, y, x+w, y, x+w, y+h, x, y+h), 1)
        
        sequence_length = len(sequence.ground_truth_rect)

        gt_x_all = sequence.ground_truth_rect[:, [0, 2, 4, 6]]
        gt_y_all = sequence.ground_truth_rect[:, [1, 3, 5, 7]]

        x1 = np.amin(gt_x_all, 1).reshape(-1,1)
        y1 = np.amin(gt_y_all, 1).reshape(-1,1)
        x2 = np.amax(gt_x_all, 1).reshape(-1,1)
        y2 = np.amax(gt_y_all, 1).reshape(-1,1)

        times = []
        tracked_bb = []
        current_frame = 0

        while current_frame < (sequence_length-1):
            # Initialize
            image = self._read_image(sequence.frames[current_frame])

            cx = np.mean(sequence.ground_truth_rect[current_frame, 0::2])
            cy = np.mean(sequence.ground_truth_rect[current_frame, 1::2])
            xmin, ymin = x1[current_frame, 0], y1[current_frame, 0]
            xmax, ymax = x2[current_frame, 0], y2[current_frame, 0]
            A1 = np.linalg.norm(sequence.ground_truth_rect[current_frame, 0:2]-sequence.ground_truth_rect[current_frame, 2:4]) * \
                 np.linalg.norm(sequence.ground_truth_rect[current_frame, 2:4]-sequence.ground_truth_rect[current_frame, 4:6])
            A2 = (xmax-xmin) * (ymax-ymin)
            S = np.sqrt(A1/A2)
            W = (xmax-xmin) * S + 1
            H = (ymax-ymin) * S + 1

            sequence.init_state = []
            sequence.init_state.append(cx-W*0.5)
            sequence.init_state.append(cy-H*0.5)
            sequence.init_state.append(W)
            sequence.init_state.append(H)

            start_time = time.time()
            self.initialize(image, sequence.init_state)
            times.append(time.time() - start_time)

            if self.params.visualization:
                self.init_visualization()
                self.visualize(image, sequence.init_state)

            # Track
            tracked_bb.append([1])
            for frame in sequence.frames[current_frame+1:]:
                current_frame += 1
                image = self._read_image(frame)

                start_time = time.time()
                state = self.track(image)
                times.append(time.time() - start_time)

                state[0] = round(state[0], 2)
                state[1] = round(state[1], 2)
                state[2] = round(state[2], 2)
                state[3] = round(state[3], 2)

                if self.params.visualization:
                    self.visualize(image, state)

                state_numpy = np.zeros((8), np.float32)
                state_numpy[0] = state[0]
                state_numpy[1] = state[1]
                state_numpy[2] = state[0] + state[2]
                state_numpy[3] = state[1]
                state_numpy[4] = state[0] + state[2]
                state_numpy[5] = state[1] + state[3]
                state_numpy[6] = state[0]
                state_numpy[7] = state[1] + state[3]

                gt_numpy = np.zeros((8), np.float32)
                gt_numpy[0] = sequence.ground_truth_rect[current_frame, 0]
                gt_numpy[1] = sequence.ground_truth_rect[current_frame, 1]
                gt_numpy[2] = sequence.ground_truth_rect[current_frame, 2]
                gt_numpy[3] = sequence.ground_truth_rect[current_frame, 3]
                gt_numpy[4] = sequence.ground_truth_rect[current_frame, 4]
                gt_numpy[5] = sequence.ground_truth_rect[current_frame, 5]
                gt_numpy[6] = sequence.ground_truth_rect[current_frame, 6]
                gt_numpy[7] = sequence.ground_truth_rect[current_frame, 7]

                poly_overlap = self.poly_iou(state_numpy, gt_numpy)
                if poly_overlap>0:
                    tracked_bb.append(state)
                else:
                    tracked_bb.append([2])
                    tracked_bb.append([0])
                    tracked_bb.append([0])
                    tracked_bb.append([0])
                    tracked_bb.append([0])
                    current_frame = current_frame + 5
                    break

        return tracked_bb, times


    def track_webcam(self):
        """Run tracker with webcam."""

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.mode_switch = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                    self.mode_switch = True
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'track'
                    self.mode_switch = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = ui_control.get_bb()
                self.initialize(frame, init_state)

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                state = self.track(frame)
                state = [int(s) for s in state]
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Put text
            font_color = (0, 0, 0)
            if ui_control.mode == 'init' or ui_control.mode == 'select':
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            elif ui_control.mode == 'track':
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()

    def visualize(self, image, state):
        self.ax.cla()
        self.ax.imshow(image)
        rect = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if self.pause_mode:
            plt.waitforbuttonpress()

    def _read_image(self, image_file: str):
        return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)

