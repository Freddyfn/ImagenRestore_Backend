class ImageRestorationService:
    def __init__(self, processor):
        self.processor = processor
    
    def process(self, image_data: bytes, method: str, tolerance: float, max_iterations: int, omega: float) -> dict:
        return self.processor.apply_filters(image_data, method, tolerance, max_iterations, omega)
