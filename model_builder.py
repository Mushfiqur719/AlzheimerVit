
import timm
import torch.nn as nn

def build_model(num_classes=4, model_name='swin_tiny_patch4_window7_224', pretrained=True, dropout_rate=0.0):
    """
    Builds a Swin Transformer model.
    """
    print(f"Building model: {model_name}")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout_rate)
    
    return model

def freeze_layers(model, freeze_backbone=False):
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze head
        if hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
            
    return model
