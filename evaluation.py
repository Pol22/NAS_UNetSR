import os
from tensorflow.keras.models import load_model
from sr_data import SR_DATA
from loss_metric import MSE, PSNR


# TODO create evaluation over all image
class Evaluator(object):
    def __init__(self, dataset_path, batch, img_size):
        self.dataset = SR_DATA(
            os.path.join(dataset_path, 'LR_bicubic', 'X2'),
            os.path.join(dataset_path,'HR')).dataset(
                batch_size=batch,
                repeat_count=4,
                random_transform=True,
                crop_size=img_size)

    def evaluate(self, models):
        '''
            Return list of pairs (psnr, model_path)
        '''
        results = list()

        for model_path in models:
            model = load_model(model_path, compile=False)
            model.compile(loss=MSE(), metrics=[PSNR()])
            result = model.evaluate(self.dataset, verbose=0)
            psnr = result[1] # (loss, PSNR) -> PSNR
            results.append((psnr, model_path))

        return results
