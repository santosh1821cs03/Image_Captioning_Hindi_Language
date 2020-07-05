from core.solver_for_corrected import CaptioningSolver
# from core.model import CaptionGenerator
# from core.model_hierarchichal import CaptionGenerator
# from core.model_top_down import CaptionGenerator
# from core.model_gru_hier import CaptionGenerator
# from core.model_gru_bahdanu import CaptionGenerator
from core.model_gru_visual import CaptionGenerator
# from core.model_adap_gru import CaptionGenerator
# from core.model_gru_luong import CaptionGenerator
from core.utils_for_corrected import load_coco_data
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

def main():
    # load train dataset
    data = load_coco_data(data_path='./data_corrected', split='train_hindi_corrected')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./data_corrected', split='val_hindi')
    
    model = CaptionGenerator(word_to_idx, dim_feature=[196,512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=16, prev2out=True, 
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=10, batch_size=128, update_rule='adam',
                                          learning_rate=4e-4, print_every=100, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='model_corrected_hindi_resnet_gru_visual-4k-val/gru/', test_model='model_corrected_hindi_resnet_gru_bahdanu_attention/gru/model-1',
                                     print_bleu=True, log_path='log/')

    solver.train()

if __name__ == "__main__":
    main()