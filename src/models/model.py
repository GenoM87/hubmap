import segmentation_models_pytorch as smp

from config import _C as cfg

def build_model(cfg):

    if cfg.MODEL.ATTENTION:
        model = smp.Unet(
            encoder_name=cfg.MODEL.NAME,
            encoder_weights=cfg.MODEL.PRETRAINING,
            in_channels=3,
            classes=1,
            decoder_attention_type='scse',
            activation='sigmoid' #attivazione sigmoid per output probabilità/ #None per far uscire i logit
        )
    else:
        model = smp.Unet(
            encoder_name=cfg.MODEL.NAME,
            encoder_weights=cfg.MODEL.PRETRAINING,
            in_channels=3,
            classes=1,
            activation='sigmoid' #attivazione sigmoid per output probabilità/ #None per far uscire i logit
        )

    return model