import cv2
import urllib.request

url = "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_russian_plate_number.xml"
filename = "haarcascade_russian_plate_number.xml"
urllib.request.urlretrieve(url, filename)

def detect_license_plate(image_path):
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    car_image = cv2.imread(image_path)
    gray = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(plates) > 0:
        x, y, w, h = plates[0]
        return x, y, w, h
    else:
        return None

def replace_license_plate_with_logo(image_path, logo_path, output_path):
    car_image = cv2.imread(image_path)

    logo = cv2.imread(logo_path)

    license_plate_area = detect_license_plate(image_path)

    if license_plate_area is not None:
        x, y, w, h = license_plate_area

        logo_resized = cv2.resize(logo, (w, h))

        car_image[y:y + h, x:x + w] = logo_resized

        cv2.imwrite(output_path, car_image)
    else:
        print("Номерной знак не обнаружен")

replace_license_plate_with_logo('car.jpg', 'logo.jpg', 'output_image.jpg')
