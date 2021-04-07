import time
from keras.preprocessing.image import load_img, img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'CN', 'DT', '/']


# define the test configuration
class CardConfig(Config):
    NAME = 'number'

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 13

    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.1


def main(path_to_weight):
    ###################################################################################
    # load this on boot
    config = CardConfig()
    # path_to_weight = 'weights/weights.h5'
    path_to_image = 'dataset/marlvinmbinga/test4.jpeg'
    rcnn = MaskRCNN(mode='inference', model_dir='./load_weights', config=config)
    rcnn.load_weights(path_to_weight, by_name=True)
    ####################################################################################

    ####################################################################################

    # When hit the endpoint does this
    img = img_to_array(load_img(path_to_image))
    results = rcnn.detect([img], verbose=1)
    r = results[0]
    ####################################################################################

    width = img.shape[1]
    xyz = zip(r['class_ids'], [list(i) for i in r['rois']], r['scores'])
    sortedXYZ = [list(i) for i in sorted(xyz, key=lambda item: item[1][1])]
    if 11 in r['class_ids']:
        CN_COD = r['rois'][list(r['class_ids']).index(11)]
        CN_COD[0] -= 20
        CN_COD[1] = 10
        CN_COD[2] += 15
        CN_COD[3] = width - 10
        res = filter(lambda x: True if CN_COD[0] < x[1][0] < CN_COD[2] else False, sortedXYZ)
        getIndexOf(res, CN_COD)
    # To see what the model is seeing comment the line below
    display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


def getIndexOf(sortedList, CN):
    print('Card Number')
    valid = [str(i) for i in range(0, 11)]
    cardNumber = ''
    prevDigit = [11, CN, 0]
    for item in sortedList:
        if str(item[0]) not in valid:
            continue
        if CN[0] < item[1][0] and item[1][2] < CN[2] and item[1][3] - item[1][1] < 500:
            overlap = lambda x, y: False if (y[1][1] - x[1][1]) / (x[1][3] - x[1][1]) > 0.7 or x[0] == 11 or y[
                0] == 11 else True
            if prevDigit is not None:
                if overlap(prevDigit, item):
                    resolve = lambda digit1, digit2: digit1 if digit1[-1] > digit2[-1] else digit2
                    candidate = resolve(prevDigit, item)
                    if prevDigit[0] == 5 and item[0] == 6 or prevDigit[0] == 6 and item[0] == 5:
                        candidate = prevDigit if prevDigit[0] == 5 else item
                    class_id = candidate[0]
                    cardNumber = cardNumber[:len(cardNumber) - 1]
                    prevDigit = candidate
                    print('overlap >> ', candidate)
                else:
                    class_id = item[0]
                    prevDigit = item
                if class_id == 10:
                    class_id = 0
                cardNumber += str(class_id)

    chunks = [cardNumber[i:i + 4] for i in range(0, len(cardNumber), 4)]
    try:
        print(f'{chunks[0]} {chunks[1]} {chunks[2]} {chunks[3]}')
    except IndexError:
        pass
    print(cardNumber)
    return cardNumber


if __name__ == '__main__':
    main()
