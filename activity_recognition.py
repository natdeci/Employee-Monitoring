from transformers import CLIPProcessor, CLIPModel

class ActivityRecognition:
    def __init__(self, prompts):
        self.prompts = prompts
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def recognize(self, image):
        inputs = self.processor(text=self.prompts, images=image, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        probabilities = outputs.logits_per_image.softmax(dim=1)
        return self.prompts[probabilities.argmax()]