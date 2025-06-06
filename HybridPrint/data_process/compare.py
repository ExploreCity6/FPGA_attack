from PIL import Image

# Load the images
image1_path = './A_2024-06-19_01-53.dat.part.png'
image2_path = './A_2024-06-21_00-45.dat.part.png'
image3_path = './A_2024-06-21_01-23.dat.part.png'

image1 = Image.open(image1_path)
image2 = Image.open(image2_path)
image3 = Image.open(image3_path)

# Concatenate images horizontally
total_width = image1.width + image2.width + image3.width
max_height = max(image1.height, image2.height, image3.height)

concatenated_image = Image.new('RGB', (total_width, max_height))
concatenated_image.paste(image1, (0, 0))
concatenated_image.paste(image2, (image1.width, 0))
concatenated_image.paste(image3, (image1.width + image2.width, 0))

# Save the concatenated image
output_path = 'compare.png'
concatenated_image.save(output_path)