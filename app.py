from flask import Flask, jsonify, request, send_file
import io
import logging
import torch
from diffusers import StableDiffusionPipeline
import warnings
import openpyxl
from openpyxl.drawing.image import Image
import os
import tempfile

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

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400

        prompt = data.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({'error': 'Prompt must be a non-empty string'}), 400

        # Generate the image
        image = generate_image(prompt)

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img_file:
            image.save(temp_img_file, format='PNG')
            temp_img_path = temp_img_file.name

        # Create an Excel file
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Generated Image"

        # Add the prompt to the first cell
        ws['A1'] = "Prompt"
        ws['B1'] = prompt

        # Embed the image in the Excel sheet
        img = Image(temp_img_path)
        ws.add_image(img, 'A2')

        # Save the Excel file to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_excel_file:
            wb.save(temp_excel_file.name)
            temp_excel_path = temp_excel_file.name

        # Clean up the temporary image file
        os.remove(temp_img_path)

        # Send the Excel file as a response
        return send_file(
            temp_excel_path,
            as_attachment=True,
            download_name='generated_image.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except ValueError as ve:
        logger.warning(f"Generation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': 'Failed to generate image'}), 500
    finally:
        # Clean up the temporary Excel file if it exists
        if 'temp_excel_path' in locals() and os.path.exists(temp_excel_path):
            try:
                os.remove(temp_excel_path)
            except Exception as e:
                logger.error(f"Failed to delete temporary Excel file: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)