import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import os

# Function to adjust image contrast using gamma correction
def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function for sinusoidal fitting
def fit_sin(time, values):
    time = np.array(time)
    values = np.array(values)
    frequency = np.fft.fftfreq(len(time), (time[1] - time[0]))
    fft_values = abs(np.fft.fft(values))
    guess_freq = abs(frequency[np.argmax(fft_values[1:]) + 1])
    guess_amp = np.std(values) * 2.**0.5
    guess_offset = np.mean(values)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sin_func(t, A, omega, phi, C): return A * np.sin(omega*t + phi) + C
    popt, pcov = curve_fit(sin_func, time, values, p0=guess)
    A, omega, phi, C = popt
    freq = omega / (2.*np.pi)
    fit_func = lambda t: A * np.sin(omega*t + phi) + C
    return {"amplitude": A, "frequency": freq, "phase": phi, "offset": C, "fit_func": fit_func, "max_cov": np.max(pcov), "raw_results": (guess, popt, pcov)}

# Configuration variables
start_frame = 450
end_frame = 500
aruco_size_mm = 57
video_path = '114.mov'
title = 'Clip #114 Empty Jar'

# Initialize OpenCV VideoCapture
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize the ArUco dictionary for 6x6 markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

vertical_positions = []
time_stamps = []
marker_heights = []

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

for frame_idx in range(start_frame, end_frame):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    frame = adjust_gamma(frame, gamma=1.5)

    if ids is not None and 2 in ids.flatten():
        id_indexes = np.where(ids.flatten() == 2)[0]
        for id_index in id_indexes:
            marker_corners = corners[id_index].reshape(-1, 2)
            center = np.mean(marker_corners, axis=0).astype(int)
            cv2.polylines(frame, [marker_corners.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.circle(frame, tuple(center), 5, (255, 0, 0), -1)

            vertical_positions.append(center[1])
            time_stamps.append(frame_idx / fps)

            left_height = np.linalg.norm(marker_corners[0] - marker_corners[3])
            right_height = np.linalg.norm(marker_corners[1] - marker_corners[2])
            average_height = (left_height + right_height) / 2
            marker_heights.append(average_height)

    cv2.imshow('Frame with Marker', frame)
    cv2.waitKey(50)

cv2.destroyAllWindows()
cap.release()

if marker_heights:
    average_marker_height = np.mean(marker_heights)
    pixels_to_mm = aruco_size_mm / average_marker_height
    print(f"Average marker height in pixels: {average_marker_height}")
else:
    print("No ArUco markers detected.")

# Convert lists to numpy arrays for analysis
time_stamps = np.array(time_stamps)
vertical_positions = np.array(vertical_positions)

if len(time_stamps) > 0 and len(vertical_positions) > 0:
    res = fit_sin(time_stamps, vertical_positions)
    amplitude = res['amplitude']
    period = 1 / res['frequency']
    phase = res['phase']
    offset = res['offset']

    time_stamps_dense = np.linspace(time_stamps.min(), time_stamps.max(), 300)
    adjusted_positions = res['fit_func'](time_stamps_dense)

    plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1])

    ax0 = plt.subplot(gs[0])
    ax0.scatter(time_stamps, -1 * (vertical_positions - offset) * pixels_to_mm, alpha=0.5, label='Observations')
    ax0.plot(time_stamps_dense, -1 * (adjusted_positions - offset) * pixels_to_mm, color='red', label='Sinusoidal Fit')
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Vertical Position (mm)')
    ax0.set_title(title)
    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), fancybox=True, shadow=True, ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    plt.subplot(gs[1]).axis('off')  # Text area
    info_template = "Amplitude px: {:.3f}\nAmplitude mm: {:.2f} mm\nPeriod: {:.4f}s\nFrequency: {:.4f}Hz\nArUco: {:.2f}mm\nmm/px: {:.2f}mm\nStart frame: {:.0f}\nEnd frame: {:.0f}\nFPS: {:.2f}"
    info_text = info_template.format(abs(amplitude), abs(amplitude * pixels_to_mm), period, 1 / period, aruco_size_mm, pixels_to_mm, start_frame, end_frame, fps)
    plt.text(0.9, 0.5, info_text, ha='center', va='center', transform=plt.gcf().transFigure)

    video_basename = os.path.splitext(video_path)[0]
    graph_filename = f"#{video_basename}.png"
    output_folder = 'png'  # Example: '/path/to/folder'
    plt.tight_layout()
    plt.savefig(graph_filename, dpi=600)
    plt.show(dpi=600)

else:
    print("No data for fitting.")
