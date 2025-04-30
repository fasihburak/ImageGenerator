from flask import Flask, jsonify, request, render_template
import io
import base64
import logging
import torch
from diffusers import StableDiffusionPipeline
import warnings

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

model_id = "runwayml/stable-diffusion-v1-5"
pipe = None
device = "cpu"

def init_model():
    """Initialize the Stable Diffusion model lazily."""
    global pipe
    if pipe is None:
        try:
            logger.info(f"Loading model {model_id} on {device}")
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
            pipe = pipe.to(device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

def generate_image(prompt: str):
    """Generate an image from a text prompt."""
    try:
        init_model()

        result = pipe(prompt, num_inference_steps=50, guidance_scale=7.5)
        image = result.images[0]

        if hasattr(result, 'nsfw_content_detected') and any(result.nsfw_content_detected):
            raise ValueError("NSFW content detected in generated image")

        return image
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        raise RuntimeError(f"Image generation failed: {str(e)}")

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400

        prompt = data.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({'error': 'Prompt must be a non-empty string'}), 400

        image = generate_image(prompt)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        encoded_image = base64.b64encode(img_byte_arr).decode('utf-8')

        return jsonify({'image': encoded_image})

    except ValueError as ve:
        logger.warning(f"Generation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': 'Failed to generate image'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
