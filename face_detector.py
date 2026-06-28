import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_face_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("could not open camera")
    print("Press 'C' to capture image"  )
    while True:
        ret , frame =cap.read()
        if not ret:
            continue
        cv2.imshow("Camera " , frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            image = frame.copy()
            cv2.imwrite("captured_image.jpg" , image)
            print("image saved as camptured_image.jpg")
            break
        if key ==27:
            cap.release()
            cv2.destroyAllWindows()
            return None
    cap.release()
    cv2.destroyAllWindows()

    h,w,_ = image.shape
    with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        refine_landmarks = True,

    ) as face_mesh:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            raise Exception("No face found")
        face_landmarks =results.multi_face_landmarks[0]
        xs =[]
        ys =[]
        for lm in face_landmarks.landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))
        x_min = max(min(xs) - 20 , 0)
        y_min = max(min(ys) - 20, 0)

        x_max = min(max(xs) + 20, w)
        y_max = min(max(ys) + 20, h)

        face_crop = image[
            y_min:y_max,
            x_min:x_max
        ]

        # Resize for your MobileNetV3 model
        face_crop = cv2.resize(
            face_crop,
            (224, 224)
        )

        cv2.imwrite(
            "face_crop.jpg",
            face_crop
        )

        print("Face crop saved as face_crop.jpg")

        return face_crop

