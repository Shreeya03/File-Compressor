import cv2
import numpy as np

# Read the RGB image
rgbImage = cv2.imread('bmw.png')

# Convert the image to grayscale
grayImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)

# Flatten the grayscale image into a vector
pixelValues = grayImage.flatten()

# Count occurrences of each pixel value
counts, uniquePixelValues = np.histogram(pixelValues, bins=range(257))

# Calculate probabilities
totalPixels = grayImage.size
prob = counts / totalPixels

# Display the counts and probabilities
print('Pixel Value   Count   Probability')

# Sort the data based on counts in descending order
sorted_data = sorted(zip(uniquePixelValues, counts, prob), key=lambda x: x[1], reverse=True)

# Print the sorted data
print("Unique Pixel Value    Count    Probability")
for i, (value, count, probability) in enumerate(sorted_data):
    print(f'{value}                {count}       {probability}')


# Convert the pixel value to a list of dictionaries
symbols = [{'Value': i, 'Probability': prob[i]} for i in range(256)]

# Sort symbols based on probability
symbols.sort(key=lambda x: x['Probability'])

# Build Huffman tree
nodes = symbols.copy()
while len(nodes) > 1:
    # Combine two nodes with lowest probability
    newNode = {'Symbol': None, 'Probability': nodes[0]['Probability'] + nodes[1]['Probability'],
               'left': nodes[0], 'right': nodes[1]}
    nodes = nodes[2:]
    nodes.append(newNode)
    nodes.sort(key=lambda x: x['Probability'])
huffTree = nodes[0]

# Generate Huffman codes
codes = [''] * 256

# Initialize stack for iterative traversal
stack = [huffTree]
codesStack = ['']

# Iterative traversal to generate Huffman codes
while stack:
    node = stack.pop()
    code = codesStack.pop()
    if 'left' in node:
        stack.append(node['left'])
        codesStack.append(code + '0')
    if 'right' in node:
        stack.append(node['right'])
        codesStack.append(code + '1')
    if 'Value' in node:
        codes[node['Value']] = code

# Assuming you have lists uniquePixelValues and codes

# Display uniquePixelValues along with their corresponding codes
# Assuming you have lists uniquePixelValues, codes, and counts

# Sort the data based on counts in descending order
sorted_data1 = sorted(zip(uniquePixelValues, codes, counts), key=lambda x: x[2], reverse=True)

# Display the sorted data
print("Index    Unique Pixel Value    Code")
for i, (pixel_value, code, count) in enumerate(sorted_data1):
    print(f'{pixel_value}                 {code}')


# Encode the image using Huffman codes
encodedImage = [[''] * grayImage.shape[1] for _ in range(grayImage.shape[0])]
for i in range(grayImage.shape[0]):
    for j in range(grayImage.shape[1]):
        pixelValue = grayImage[i, j]
        encodedImage[i][j] = codes[pixelValue]

# print(encodedImage)

encodedImagec = [''.join(row) for row in encodedImage]

# print(encodedImagec)
with open("encoded.txt", 'w') as f:
    for i in encodedImagec:
        f.write(i)
#bpsk modulation

def bpsk_modulation(binary_strings):
    modulation = []
    for string in binary_strings:
        signal = []
        for bit in string:
            if bit == '0':
                signal.append(-1)  # Represent 0 with -1 phase
            else:
                signal.append(1)   # Represent 1 with +1 phase
        modulation.append(signal)
    return modulation

modulatedimage=bpsk_modulation(encodedImagec)

# for i in modulatedimage:
#     print(i)
#     print('\n')

with open('modulated_image.txt', 'w') as file:
    for signal in modulatedimage:
        file.write(' '.join(map(str, signal)) + '\n')





def add_white_gaussian_noise(signal, snr_dB):
   
    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Calculate noise power using SNR in dB
    noise_power = signal_power / (10 ** (snr_dB / 10))

    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # Add noise to the signal
    noisy_signal = np.array(signal) + noise

    return noisy_signal

# Example usage with a signal-to-noise ratio (SNR) of 10 dB
snr_dB = -10
noisy_modulated_image = [add_white_gaussian_noise(signal, snr_dB) for signal in modulatedimage]

# print(noisy_modulated_image)

#demodulation after addition of noise
def bpsk_demodulation(modulated_signal):
    
    demodulated_bits = []
    threshold = 0  # Threshold for demodulation

    for signal_sample in modulated_signal:
        if signal_sample > threshold:
            demodulated_bits.append('1')
        else:
            demodulated_bits.append('0')

    return demodulated_bits

# Demodulate the noisy modulated image
demodulated_bits = [bpsk_demodulation(signal) for signal in noisy_modulated_image]

# print(demodulated_bits)
# concatenated_bits = []
# for sublist in demodulated_bits:
#     concatenated_bits.extend(sublist)



# Now, concatenated_bits contains all the bits in a single list

demodulated_bits = [''.join(row) for row in demodulated_bits]
# print(demodulated_bits)

print(demodulated_bits)
# concatenated_bits = []
# for sublist in demodulated_bits:
#     concatenated_bits.extend(sublist)



# decodedImage = np.zeros_like(grayImage)
reverseCodes = {code: value for value, code in enumerate(codes)}

# Now each row in demodulated_bits is the same size as the corresponding row in encodedImage
decodedImage=[]
# Iterate through each encoded row
for i, row in enumerate(demodulated_bits):
    pixelValues = []
    code = ''
    for bit in row:
        code += bit
        if code in reverseCodes:
            pixelValues.append(reverseCodes[code])
            code = ''
    # Assign decoded pixel values to the corresponding row
    decodedImage.append(pixelValues)
    # decodedImage[i] = np.array(pixelValues).reshape(1, -1)



# Convert decodedImage_padded to a NumPy array
max_length = max(len(row) for row in decodedImage)

# Pad or truncate each row to have the same length
for i in range(len(decodedImage)):
    decodedImage[i] = decodedImage[i][:max_length] + [1] * (max_length - len(decodedImage[i]))


decodedImage_np = np.array(decodedImage, dtype=np.uint8)
print(decodedImage_np)
# print(decodedImage_np.shape)

import numpy as np
import cv2
import matplotlib.pyplot as plt

# SNR values from -10 dB to 10 dB with a 3 dB gap
snr_values = np.arange(-10, 15, 1)

# Error array to store the error for each SNR value
errors = []

for snr_dB in snr_values:
    # Add white Gaussian noise to the modulated image
    noisy_modulated_image = [add_white_gaussian_noise(signal, snr_dB) for signal in modulatedimage]
    
    # Demodulate the noisy modulated image
    demodulated_bitss = [bpsk_demodulation(signal) for signal in noisy_modulated_image]
    
    # Convert demodulated_bits to a flat list
    demodulated_bitss = [''.join(row) for row in demodulated_bitss]
    
    # Calculate the error: count the bits that are different between encodedImagec and demodulated_bits
    error = sum(1 for x, y in zip(encodedImagec, demodulated_bitss) if x != y)
    
    # Append the error to the errors list
    errors.append(error)

# Plot the graph for error versus SNR in log scale
plt.figure()
plt.plot(snr_values, errors, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('Error (Number of Bits)')
plt.title('Error vs SNR')
# plt.yscale('log')
plt.grid(True)
plt.show()

# decoded image
cv2.imshow('Decoded Image', decodedImage_np)



cv2.waitKey(0)
cv2.destroyAllWindows()