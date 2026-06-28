import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

FACE_OVAL = [
    10,338,297,332,284,251,389,356,454,323,
    361,288,397,365,379,378,400,377,152,148,
    176,149,150,136,172,58,132,93,234,127,
    162,21,54,103,67,109
]

LEFT_EYE = [
    33,7,163,144,145,153,154,155,
    133,173,157,158,159,160,161,246
]

RIGHT_EYE = [
    362,382,381,380,374,373,390,249,
    263,466,388,387,386,385,384,398
]

LIPS = [
    61,146,91,181,84,17,314,405,
    321,375,291,308,324,318,402,317,
    14,87,178,88,95,78
]


def segment_skin(face_crop):

    h, w = face_crop.shape[:2]

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        rgb = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return face_crop

        face = results.multi_face_landmarks[0]

        mask = np.zeros((h, w), dtype=np.uint8)

        # Face oval
        face_pts = np.array([
            (
                int(face.landmark[idx].x * w),
                int(face.landmark[idx].y * h)
            )
            for idx in FACE_OVAL
        ], dtype=np.int32)

        cv2.fillPoly(mask, [face_pts], 255)

        # Remove left eye
        left_eye_pts = np.array([
            (
                int(face.landmark[idx].x * w),
                int(face.landmark[idx].y * h)
            )
            for idx in LEFT_EYE
        ], dtype=np.int32)

        cv2.fillPoly(mask, [left_eye_pts], 0)

        # Remove right eye
        right_eye_pts = np.array([
            (
                int(face.landmark[idx].x * w),
                int(face.landmark[idx].y * h)
            )
            for idx in RIGHT_EYE
        ], dtype=np.int32)

        cv2.fillPoly(mask, [right_eye_pts], 0)

        # Remove lips
        lip_pts = np.array([
            (
                int(face.landmark[idx].x * w),
                int(face.landmark[idx].y * h)
            )
            for idx in LIPS
        ], dtype=np.int32)

        cv2.fillPoly(mask, [lip_pts], 0)

        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        skin_only = cv2.bitwise_and(
            face_crop,
            face_crop,
            mask=mask
        )

        return skin_only