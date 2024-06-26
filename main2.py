import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import os

# Fonction pour ajuster le contraste par la correction gamma
def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Fonction pour ajustement sinusoidal
def fit_sin(tt, yy):
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c): return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amplitude": A, "frequency": f, "phase": p, "offset": c, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess, popt, pcov)}

# Variables de configuration
start_img = 450
end_img = 500
aruco_real = 57
video_path = '114.mov'
titre = 'Clip #114 Bocal Vide'
aruco_id = 2
display = 0

# Initialiser OpenCV VideoCapture
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_in_seconds = total_frames / cap.get(cv2.CAP_PROP_FPS)
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialiser le dictionnaire ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

positions = []
timestamps = []
hauteurs = []

cap.set(cv2.CAP_PROP_POS_FRAMES, start_img)

for frame_idx in range(start_img, end_img):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    frame = adjust_gamma(frame, gamma=1.5)

    if ids is not None and aruco_id in ids.flatten():
        id_indexes = np.where(ids.flatten() == aruco_id)[0]
        for id_index in id_indexes:
            marker_corners = corners[id_index].reshape(-1, 2)
            center = np.mean(marker_corners, axis=0).astype(int)
            cv2.polylines(frame, [marker_corners.astype(np.int32)], True, (0, 255, 255), 2)
            cv2.circle(frame, tuple(center), 5, (255, 0, 0), -1)

            positions.append(center[1])
            timestamps.append(frame_idx / fps)

            hauteur_gauche = np.linalg.norm(marker_corners[0] - marker_corners[3])
            hauteur_droite = np.linalg.norm(marker_corners[1] - marker_corners[2])
            hauteur_moyenne = (hauteur_gauche + hauteur_droite) / 2
            hauteurs.append(hauteur_moyenne)

    if display == 1:
        cv2.imshow('Frame avec marqueur', frame)
        cv2.waitKey(50)

if display == 1:
    cv2.destroyAllWindows()
cap.release()

# Reste du code de traitement et affichage des résultats...

if hauteurs:
    hauteur_moyenne_totale = np.mean(hauteurs)
    pix_to_mm = aruco_real / hauteur_moyenne_totale
    print(f"Hauteur moyenne du marqueur ArUco en pixels : {hauteur_moyenne_totale}")
else:
    print("Aucun marqueur ArUco détecté.")

# Convertir les listes en arrays NumPy pour l'analyse
timestamps = np.array(timestamps)
positions = np.array(positions)

# Vérifier que nous avons des données pour éviter des erreurs lors de l'appel de fit_sin
if len(timestamps) > 0 and len(positions) > 0:
    # Appliquer l'ajustement sinusoïdal en utilisant la fonction 'fit_sin'

    res = fit_sin(timestamps, positions)
    amplitude = res['amplitude']
    periode = 1 / res['frequency']
    phase = res['phase']
    offset = res['offset']

    # Générer des timestamps denses pour une courbe lisse
    timestamps_dense = np.linspace(timestamps.min(), timestamps.max(), 300)

    # Calculer les positions ajustées avec la fonction sinusoidale
    positions_ajustees = res['fitfunc'](timestamps_dense)

    # Créer une figure et un GridSpec avec deux colonnes
    plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1])  # 3:1 ratio pour graphique : zone texte

    # Créer l'axe pour le graphique
    ax0 = plt.subplot(gs[0])
    ax0.scatter(timestamps, -1*(positions-offset)*pix_to_mm, alpha=0.5, label='Observations')
    ax0.plot(timestamps_dense, -1*(positions_ajustees-offset)*pix_to_mm, color='red', label='Ajustement sinusoïdal')
    ax0.set_xlabel('Temps (s)')
    ax0 = plt.gca()  # Obtient l'objet axe courant
    ax0.xaxis.set_major_locator(ticker.MaxNLocator(20))

    ax0.set_ylabel('Position verticale (mm)')
    ax0.set_title(titre)

    # Placer la légende en bas sous l'axe des X
    # Ajustez 'bbox_to_anchor' selon la nécessité pour le placement précis
    ax0.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), fancybox=True, shadow=True, ncol=2)

    # Ajustez la mise en page pour éviter que la légende ou d'autres éléments soient coupés
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Ajustez la marge du bas si nécessaire
    #plt.gca().invert_yaxis()

    # Zone pour le texte
    plt.subplot(gs[1])  # Active la seconde partie du GridSpec pour le texte

    # Utiliser figtext pour ajouter le texte à l'extérieur du graphique
    info_template = "Amplitude px: {:.3f}\nAmplitude mm: {:.2f} mm\nPériode: {:.4f}s\nFréquence: {:.4f}Hz\nArUco : {:.2f}mm\nmm/px:{:.2f}mm\nStart img :{:.0f}\nEnd img :{:.0f}\nips :{:.2f}"
    info_text = info_template.format(abs(amplitude),  abs(amplitude*pix_to_mm), periode,1/periode, aruco_real,pix_to_mm,start_img,end_img,fps)

    # Positionner le texte dans la zone réservée
    plt.axis('off')  # Masquer les axes pour la zone de texte
    plt.text(0.9, 0.5, info_text, ha='center', va='center', transform=plt.gcf().transFigure)

    # Enregistrer graphique au bon endroit avec le nom de la video
    video_basename = os.path.splitext(video_path)[0]
    graph_filename = f"#{video_basename}.png"
    output_folder = 'png'  # Exemple : '/chemin/vers/dossier'
    plt.tight_layout()
    plt.savefig(graph_filename, dpi=600)
    plt.show(dpi=600)

else:
    print("Aucune donnée pour l'ajustement.")
