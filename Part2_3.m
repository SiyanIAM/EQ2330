
% ========== 2.3 Distortion and Bit-Rate Estimation ==========

% Parameters
block_size = 8;
step_sizes = 2.^(0:9);
dct_matrix = dctmtx(block_size);

% Load images (replace with your image loading process)
image_boat = double(imread("images/boats512x512.tif"));
image_harbour = double(imread("images/harbour512x512.tif"));
image_peppers = double(imread("images/peppers512x512.tif"));
images = {image_boat, image_harbour, image_peppers};

% Process images and calculate PSNR/bitrate
[bitrates, psnrs] = process_images(images, step_sizes, block_size, dct_matrix);

% Average PSNR and bitrate
average_psnrs = mean(psnrs, 2);
average_bitrates = mean(bitrates, 2);

% Plot Rate-PSNR Curve
figure('Name', 'Rate-PSNR Curve');
plot(average_bitrates, average_psnrs, '-bo');
xlabel("Bit-rates");
ylabel("PSNR (dB)");
legend("DCT");
grid on;

function [quantized_blocks, reconstructed_image] = process_image_blocks(image, step_size, block_size, dct_matrix)
    % Process a single image: DCT, quantization, and reconstruction
    [rows, cols] = size(image);
    quantized_blocks = zeros(size(image));
    reconstructed_image = zeros(size(image));
    for row = 1:block_size:rows
        for col = 1:block_size:cols
            % Extract block
            block = image(row:row+block_size-1, col:col+block_size-1);
            
            % DCT
            dct_block = dct2(block);
            
            % Quantization
            quantized_block = uniform_quantizer(dct_block, step_size);
            
            % Store quantized block
            quantized_blocks(row:row+block_size-1, col:col+block_size-1) = quantized_block;
            
            % Reconstruct using iDCT
            reconstructed_image(row:row+block_size-1, col:col+block_size-1) = ...
                dct_matrix' * quantized_block * dct_matrix;
        end
    end
end

function [bitrate_values, psnr_values] = process_images(images, step_sizes, block_size, dct_matrix)
    % Process multiple images for given step sizes
    num_images = numel(images);
    num_steps = length(step_sizes);
    bitrate_values = zeros(num_steps, num_images);
    psnr_values = zeros(num_steps, num_images);

    for idx = 1:num_steps
        step_size = step_sizes(idx);
        for img_idx = 1:num_images
            image = images{img_idx};
            
            % Process blocks
            [quantized_blocks, reconstructed_image] = process_image_blocks(image, step_size, block_size, dct_matrix);
            
            % Calculate bitrate and PSNR
            bitrate_values(idx, img_idx) = bitrate(quantized_blocks);
            psnr_values(idx, img_idx) = psnr8(image, reconstructed_image);
        end
    end
end

function output = mse(img1, img2)
    % Mean Squared Error (MSE) calculation
    [M, N] = size(img1);
    output = sum((img1(:) - img2(:)).^2) / (M * N);
end


function psnr_value = psnr8(original_image, reconstructed_image)
    % Peak Signal-to-Noise Ratio (PSNR) calculation
    mean_square_error = mse(original_image, reconstructed_image);
    max_pixel_value = 255; % For 8-bit images
    psnr_value = 10 * log10((max_pixel_value^2) / mean_square_error);
end

function output = uniform_quantizer(input, step_size)
    % Uniform quantization
    output = step_size * round(input / step_size);
end

function entropy = bitrate(image)
 
    flattened = round(image(:)); % Ensure values are integers
    min_val = min(flattened);    % Find the minimum value in the array
    
    % Shift values to make them positive indices (for accumarray)
    if min_val < 0
        flattened = flattened - min_val + 1;
    else
        flattened = flattened + 1;
    end

    % Calculate frequencies of each unique value
    max_val = max(flattened); 
    pixel_frequencies = accumarray(flattened, 1, [max_val, 1]);

    % Calculate probabilities
    pixel_probabilities = pixel_frequencies / numel(flattened);

    % Remove zero probabilities and compute entropy
    non_zero_probabilities = pixel_probabilities(pixel_probabilities > 0);
    entropy = -sum(non_zero_probabilities .* log2(non_zero_probabilities));
end

