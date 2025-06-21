# Test Images for Facial Recognition

This directory contains test images organized by person for facial recognition testing.

## Directory Structure

```
test_images/
├── person2/          # Images for María García
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
└── person3/          # Images for Carlos López
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg
```

## Student Image Sources

The `init_data.py` script creates 4 test students with the following image sources:

1. **Juan Pérez** (`juan.perez@test.com`)
   - 1 mock image only
   - User will add real images from frontend for testing

2. **María García** (`maria.garcia@test.com`)
   - Real images from `test_images/person2/` folder
   - Up to 3 images will be processed

3. **Carlos López** (`carlos.lopez@test.com`)
   - Real images from `test_images/person3/` folder
   - Up to 3 images will be processed

4. **Ana Martínez** (`ana.martinez@test.com`)
   - Uses existing images from `media/student_faces/`
   - Falls back to mock if no existing images

## How to Use

1. **Add Images**: Place 2-3 photos in the respective folders:
   - `person2/` for María García
   - `person3/` for Carlos López
   - Use clear, front-facing photos with good lighting

2. **Run Init Data**: Execute `python init_data.py` to:
   - Load and process real images through facial recognition pipeline
   - Generate real embeddings (not mock data)
   - Associate them with the corresponding test users

3. **Test Recognition**: Use these students to test:
   - Juan: Test frontend image upload functionality
   - María & Carlos: Test recognition with real embeddings
   - Ana: Test with existing media images

## Notes

- If no images are found in folders, the system falls back to mock data
- Images should contain only one clear face per image
- Supported formats: JPG, JPEG, PNG
- The system automatically detects faces and generates embeddings 