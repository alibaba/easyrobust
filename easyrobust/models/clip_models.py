import torch
from tqdm import tqdm
from timm.models.registry import register_model
from easyrobust.third_party.clip import clip
from easyrobust.third_party.clip import names_templates

class CLIP_ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

class CLIP_ImageClassifier(torch.nn.Module):
    def __init__(self, model_name, class_names=names_templates.openai_imagenet_classnames, templates=names_templates.openai_imagenet_template):
        super().__init__()
        self.class_names = class_names
        self.templates = templates
        self.clip_model, self.train_preprocess, self.val_preprocess = clip.load(model_name, jit=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classification_head = self.get_zeroshot_weights()

        if hasattr(self.clip_model, 'transformer'):
            delattr(self.clip_model, 'transformer')

    def get_zeroshot_weights(self):
        logit_scale = self.clip_model.logit_scale
        self.clip_model.eval()
        self.clip_model.to(self.device)

        print('Getting zeroshot weights.')
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.class_names):
                texts = []
                for t in self.templates:
                    texts.append(t(classname))
                texts = clip.tokenize(texts).to(self.device) # tokenize
                embeddings = self.clip_model.encode_text(texts) # embed with text encoder
                embeddings /= embeddings.norm(dim=-1, keepdim=True)

                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings /= embeddings.norm()

                zeroshot_weights.append(embeddings)

            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(self.device)
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

            zeroshot_weights *= logit_scale.exp()
            
            zeroshot_weights = zeroshot_weights.squeeze().float()
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

        classification_head = CLIP_ClassificationHead(normalize=True, weights=zeroshot_weights)

        return classification_head

    def forward(self, inputs):
        outputs = self.classification_head(self.clip_model.encode_image(inputs))
        return outputs


@register_model
def clip_resnet50(**kwargs):
    model = CLIP_ImageClassifier('RN50')
    return model

@register_model
def clip_resnet101(**kwargs):
    model = CLIP_ImageClassifier('RN101')
    return model

@register_model
def clip_vit_base_patch32_224(**kwargs):
    model = CLIP_ImageClassifier('ViT-B/32')
    return model

@register_model
def clip_vit_base_patch16_224(**kwargs):
    model = CLIP_ImageClassifier('ViT-B/16')
    return model

@register_model
def clip_vit_large_patch14_224(**kwargs):
    model = CLIP_ImageClassifier('ViT-L/14')
    return model

@register_model
def clip_vit_large_patch14_336(**kwargs):
    model = CLIP_ImageClassifier('ViT-L/14@336px')
    return model
