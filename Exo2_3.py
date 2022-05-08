from math import ceil
import time
from random import uniform, sample

import cv2
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch



def detection_coin_FAST(image, centre, seuil):
    """
        où les arguments en entrée sont :
        — image une image (en noir et blanc) ;
        — centre un vecteur donnant la coordonnée x-y pour laquelle 
        tester la présence d’un coin dans l’ image ;
        — seuil le seuil t, tel que spécifié dans les équations de 
        l’acétate 149 de 03-VisionII.pdf ;
        
        Retourne:
        — is_fast_corner indique la présence (1) ou l’absence (0) d’un coin ;
        — intensite_coin donne l’intensité du coin représenté par la somme V , 
        telle que spécifiée à l’acétate 150 ;
    """
    is_fast_corner = False
    intensite_coin = 0

    # Tous les points sur l'arc de cercle
    points = [
        (centre[0], centre[1]-3),
        (centre[0]+1, centre[1]-3),
        (centre[0]+2, centre[1]-2),
        (centre[0]+3, centre[1]-1),
        (centre[0]+3, centre[1]),
        (centre[0]+3, centre[1]+1),
        (centre[0]+2, centre[1]+2),
        (centre[0]+1, centre[1]+3),
        (centre[0], centre[1]+3),
        (centre[0]-1, centre[1]+3),
        (centre[0]-2, centre[1]+2),
        (centre[0]-3, centre[1]+1),
        (centre[0]-3, centre[1]),
        (centre[0]-3, centre[1]-1),
        (centre[0]-2, centre[1]-2),
        (centre[0]-1, centre[1]-3),
    ]

    Ip = int(image[centre])

    # Déterminer intensite_coin:
    for x in points: 
        Ix = int(image[x])
        intensite_coin += abs(Ix - Ip)


    # Déterminer is_fast_corner:
    count = 0

    ## condition pour 1 et 9
    Ix = int(image[points[0]])
    if abs(Ix - Ip) > seuil:
        count += 1
    
    Ix = int(image[points[8]])
    if abs(Ix - Ip) > seuil:
        count += 1

    if count < 2:
        return is_fast_corner, intensite_coin

    ## condition pour 5 et 13
    Ix = int(image[points[4]])
    if abs(Ix - Ip) > seuil:
        count += 1
    
    Ix = int(image[points[12]])
    if abs(Ix - Ip) > seuil:
        count += 1

    if count < 3:
        return is_fast_corner, intensite_coin

    ## condition pour le reste des points:
    for index, x in enumerate(points):
        if index not in [0, 4, 8, 12]:
            Ix = int(image[points[12]])
            if abs(Ix - Ip) > seuil:
                count += 1
    is_fast_corner = (count >= 12)
    return is_fast_corner, intensite_coin


def ExtractBRIEF(ImagePatch, BriefDescriptorConfig):
    """
        où les arguments d'entrées sont:
        — ImagePatch est une patch d'une image noir et blanc de SxS pixels;
        — BriefDescriptorConfig est une structure de données

        Retourne un descripteur sous forme de liste de booléen (True ou False)
    """
    descriptor = []
    for x, y in BriefDescriptorConfig:
        descriptor.append(ImagePatch[x] < ImagePatch[y])
        
    return descriptor


def detect_and_extract(img):
    r = 8 # Retirer les pixels qui ne nous intéresse pas

    corners = []

    imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    imgray = cv2.GaussianBlur(imgray, (9,9), cv2.BORDER_DEFAULT)

    # Extraction des coins
    for i in range(r, imgray.shape[0]-r):
        for j in range(r, imgray.shape[1]-r):
            is_corner, intensite_coin = detection_coin_FAST(imgray, (i, j), 10)
            if is_corner:
                corners.append({ "point":(i, j), "intensite_coin": intensite_coin })
            

    print("Nombre de coins: %d, pourcentage: %.2f" % (len(corners), len(corners)*100 / (imgray.size - 4*r)))

    # counts, bins = np.histogram([corner["intensite_coin"] for corner in corners])
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.show()

    # Suppression des non-maxima locaux
    index = 1
    while index < len(corners):
        if distance.euclidean(corners[index-1]["point"], corners[index]["point"]) <= 4:
            if corners[index]["intensite_coin"] > corners[index-1]["intensite_coin"]:
                del(corners[index-1])
            else:
                del(corners[index])
            continue
        index += 1

    print("Nombre de coins: %d, pourcentage: %.2f" % (len(corners), len(corners)*100 / (imgray.size - 4*r)))

    # Sélection des coins les plus forts
    corners.sort(key=(lambda corner: corner["intensite_coin"]))
    corners = corners[:ceil(len(corners)/10)]   
        
    # Extraction des descripteurs    
    for index, corner in enumerate(corners):
        point = corner["point"]
        ImagePatch = imgray[point[0]-S//2:point[0]+S//2, point[1]-S//2:point[1]+S//2]
        corners[index]["BRIEF_descriptor"] = ExtractBRIEF(ImagePatch, BriefDescriptorConfig)

    return corners


def brute_force_match(left_corners, right_corners):
    matches = []
    for cornerL in left_corners:
        l = []
        for cornerR in right_corners:
            l.append({
                "point1": cornerL["point"], 
                "point2": cornerR["point"],
                "dist": distance.hamming(
                    cornerL["BRIEF_descriptor"], 
                    cornerR["BRIEF_descriptor"]
                )
            })

        l.sort(key=(lambda corner: corner["dist"]))
        matches.append(l[0])

    return matches


def mutually_best_match(left_corners, right_corners):
    iMatches = []
    for cornerL in left_corners:
        l = []
        for cornerR in right_corners:
            l.append({
                "point1": cornerL["point"], 
                "point2": cornerR["point"],
                "dist": distance.hamming(
                    cornerL["BRIEF_descriptor"], 
                    cornerR["BRIEF_descriptor"]
                )
            })

        l.sort(key=(lambda corner: corner["dist"]))
        iMatches.append(l[0])
    
    jMatches = []
    for cornerR in right_corners:
        r = []
        for cornerL in left_corners:
            r.append({
                "point1": cornerL["point"], 
                "point2": cornerR["point"],
                "dist": distance.hamming(
                    cornerL["BRIEF_descriptor"], 
                    cornerR["BRIEF_descriptor"]
                )
            })

        r.sort(key=(lambda corner: corner["dist"]))
        jMatches.append(l[0])

    matches = []
    for iMatch in iMatches:
        for jMatch in jMatches:
            if iMatch["point1"] == jMatch["point1"] and iMatch["point2"] == jMatch["point2"]:
                matches.append(iMatch)

    return matches


def lowe_match(left_corners, right_corners):
    matches = []
    for cornerL in left_corners:
        l = []
        for cornerR in right_corners:
            l.append({
                "point1": cornerL["point"], 
                "point2": cornerR["point"],
                "dist": distance.hamming(
                    cornerL["BRIEF_descriptor"], 
                    cornerR["BRIEF_descriptor"]
                )
            })

        l.sort(key=(lambda corner: corner["dist"]))
        if l[1]["dist"] != 0 and (l[0]["dist"] / l[1]["dist"]) < 0.6:
            matches.append(l[0])

    return matches


def plot_img_with_corners(img, corners, ax, title):
    ax.imshow(img, cmap="gray")
    for corner in corners:
        point = corner["point"]
        ax.scatter(point[1], point[0], facecolors='none', edgecolors='r', s=16)
    ax.title.set_text(title)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def plot_matches(matches, fig, ax1, ax2):
    for match in matches:
        point1, point2 = match["point1"], match["point2"]
        con = ConnectionPatch(xyA=[point1[1], point1[0]], xyB=[point2[1], point2[0]], coordsA="data", coordsB="data", axesA=ax1, axesB=ax2, color="green")
        fig.add_artist(con)
    

if __name__ == "__main__":  
    imgL = cv2.imread('./source/bw-rectified-left-022148small.png')
    imgR = cv2.imread('./source/bw-rectified-right-022148small.png')

    S = 15
    numberOfBits = 200
    BriefDescriptorConfig = []

    wRand = lambda n: ceil(uniform(-n/2, n/2))

    while(len(BriefDescriptorConfig) < (numberOfBits/8)):
        x = wRand(S), wRand(S)
        y = wRand(S), wRand(S)
        BriefDescriptorConfig.append((x, y))

    left_corners = detect_and_extract(imgL)
    right_corners = detect_and_extract(imgR)     

    fig, ax = plt.subplots(1, 2, constrained_layout=True)

    plot_img_with_corners(imgL, left_corners, ax[0], "Left")
    plot_img_with_corners(imgR, right_corners, ax[1], "Right")

    # Appariement images gauche et droite brute force:
    # matches = brute_force_match(left_corners, right_corners)

    # Mutually-best Match
    # matches = mutually_best_match(left_corners, right_corners)    

    # Test de Lowe
    matches = lowe_match(left_corners, right_corners)

    if (len(matches) > 100):
        matches = sample(matches, 100)

    plot_matches(matches[:20], fig, ax[0], ax[1])

    plt.show()
