images = cell(1, 3);
images{1} = 255 * im2double(imread("images/boats512x512.tif"));
images{2} = 255 * im2double(imread("images/harbour512x512.tif"));
images{3} = 255 * im2double(imread("images/peppers512x512.tif"));

num_images = numel(images);
num_scales = 4;
step_sizes = 2.^(0:9);
num_steps = numel(step_sizes);

bitrates_per_step = zeros(num_steps, 4, num_scales);
average_bitrates = zeros(num_steps, 1);
psnr_values = zeros(num_steps, num_images);

for step_index = 1:num_steps
    concatenated_coefficients = cell(4, num_scales);
    for image_index = 1:num_images
        % Apply Forward Wavelet Transform (FWT) to each image
        [lowpass_coeffs, hori_coeffs, vert_coeffs, diag_coeffs] = fwt(images{image_index}, num_scales);
        
        % Quantize
        for scale = 1:num_scales
            lowpass_coeffs{scale} = quantize_coeffs(lowpass_coeffs{scale}, step_sizes(step_index));
            hori_coeffs{scale} = quantize_coeffs(hori_coeffs{scale}, step_sizes(step_index));
            vert_coeffs{scale} = quantize_coeffs(vert_coeffs{scale}, step_sizes(step_index));
            diag_coeffs{scale} = quantize_coeffs(diag_coeffs{scale}, step_sizes(step_index));
        end

        % Reconstruct the image using Inverse FWT
        reconstructed_image = ifwt(lowpass_coeffs, hori_coeffs, vert_coeffs, diag_coeffs, num_scales);

        % Calculate PSNR
        psnr_values(step_index, image_index) = psnr8(images{image_index}, reconstructed_image);

        % Merge coefficients for bitrate calculation
        for scale_index = 1:num_scales
            concatenated_coefficients{1, scale_index} = [concatenated_coefficients{1, scale_index}; lowpass_coeffs{scale_index}(:)];
            concatenated_coefficients{2, scale_index} = [concatenated_coefficients{2, scale_index}; hori_coeffs{scale_index}(:)];
            concatenated_coefficients{3, scale_index} = [concatenated_coefficients{3, scale_index}; vert_coeffs{scale_index}(:)];
            concatenated_coefficients{4, scale_index} = [concatenated_coefficients{4, scale_index}; diag_coeffs{scale_index}(:)];
        end
    end

    % Calculate bitrate for each subband
    for subband_index = 1:4
        for scale_index = 1:num_scales
            bitrates_per_step(step_index, subband_index, scale_index) = bitrate(concatenated_coefficients{subband_index, scale_index});
        end
    end

    % Calculate weighted average bitrate
    weights = 1 ./ (4 .^ (1:num_scales));
    for scale_index = 1:num_scales
        average_bitrates(step_index) = average_bitrates(step_index) + weights(scale_index) * sum(bitrates_per_step(step_index, 2:4, scale_index));
    end
    average_bitrates(step_index) = average_bitrates(step_index) + weights(end) * bitrates_per_step(step_index, 1, num_scales);
end

% Calculate average PSNR across images
average_psnr = mean(psnr_values, 2).';

% Plot results
figure;
plot(average_bitrates, average_psnr, '-ro');
xlabel("Bitrates");
ylabel("PSNR");
legend("FWT");
grid on;


function quantized_coeffs = quantize_coeffs(coeffs, step_size)
    % Quantize the coefficients directly
    quantized_coeffs = step_size * round(coeffs / step_size);
end

function [approx_coeffs, hori_coeffs, verti_coeffs, diag_coeffs] = fwt(img, num_scales)
    [LPF, ~, ~, ~] = wfilters("db8");
    HPF = wrev(qmf(wrev(LPF)));

    current_image = img;
    approx_coeffs = cell(1, num_scales);
    hori_coeffs = cell(1, num_scales);
    verti_coeffs = cell(1, num_scales);
    diag_coeffs = cell(1, num_scales);

    % Achieve scaling by recursively applying the filter
    for scale = 1:num_scales
        img_size = size(current_image, 1);
        lowpass_temp = zeros([img_size, img_size / 2]);
        highpass_temp = zeros([img_size, img_size / 2]);

        % horizontal direction
        for x = 1:img_size
            [lowpass_temp(x, :), highpass_temp(x, :)] = anal_filter(current_image(x, :), LPF, HPF);
        end

        approx = zeros([img_size / 2, img_size / 2]);
        hori = zeros([img_size / 2, img_size / 2]);
        verti = zeros([img_size / 2, img_size / 2]);
        diag = zeros([img_size / 2, img_size / 2]);

        % vertical direction
        for y = 1:img_size / 2
            [approx(:, y), hori(:, y)] = anal_filter(lowpass_temp(:, y).', LPF, HPF);
            [verti(:, y), diag(:, y)] = anal_filter(highpass_temp(:, y).', LPF, HPF);
        end

        approx_coeffs{scale} = approx;
        hori_coeffs{scale} = hori;
        verti_coeffs{scale} = verti;
        diag_coeffs{scale} = diag;
        current_image = approx; 
    end

    function [approx, detail] = anal_filter(signal, LPF, HPF)
        % Add the periodic extension to the input signal to prevent border effect
        padded_signal = padarray(signal, [0, length(LPF) - 1], "circular");
        lowpass_res = conv(padded_signal, LPF, "full");
        highpass_res = conv(padded_signal, HPF, "full");

        lowpass_res = lowpass_res((length(LPF) - 1) * 2:(length(LPF) - 1) * 2 + length(signal) - 1);
        highpass_res = highpass_res((length(HPF) - 1) * 2:(length(HPF) - 1) * 2 + length(signal) - 1);

        approx = downsample(lowpass_res, 2);
        detail = downsample(highpass_res, 2);
    end
end

function output = ifwt(approx_coeffs, hori_coeffs, vert_coeffs, diag_coeffs, num_scales)
    [ ~, ~, LPF, ~] = wfilters("db8");
    HPF = qmf(LPF);
    output = approx_coeffs{num_scales};

    for scale = num_scales:-1:1
        approx = output;
        img_length = length(approx);
        hori = hori_coeffs{scale};
        vert = vert_coeffs{scale};
        diag = diag_coeffs{scale};

        temp_lp = zeros(size(approx, 1) * 2, size(approx, 2));
        temp_hp = zeros(size(approx, 1) * 2, size(approx, 2));

        % vertical direction
        for y = 1:img_length
            temp_lp(:, y) = syn_filter(approx(:, y).', hori(:, y).', LPF, HPF);
            temp_hp(:, y) = syn_filter(vert(:, y).', diag(:, y).', LPF, HPF);
        end

        % horizontal direction
        output = zeros([img_length*2, img_length*2]);
        for x = 1:img_length*2
            output(x, :) = syn_filter(temp_lp(x, :), temp_hp(x, :), LPF, HPF);
        end
    end
end

function output = syn_filter(approx, detail, LPF, HPF)
    % Upsample approximation and detail coefficients
    upsampled_approx = upsample(approx, 2);
    upsampled_dtl = upsample(detail, 2);

    % Add the periodic extension to the input signal to prevent border effect
    padded_approx = padarray(upsampled_approx, [0, length(LPF) - 1], "circular");
    padded_dtl = padarray(upsampled_dtl, [0, length(HPF) - 1], "circular");

    % Convolve with low-pass and high-pass filters
    approx_rec = conv(padded_approx, LPF, "full");
    dtl_rec = conv(padded_dtl, HPF, "full");

    approx_rec = approx_rec(length(LPF) + 1:end - (length(LPF) - 1) * 2 + 1);
    dtl_rec = dtl_rec(length(HPF) + 1:end - (length(HPF) - 1) * 2 + 1);

    output = approx_rec + dtl_rec;
end

function psnr_value = psnr8(original_image, reconstructed_image)
    % Peak PSNR calculation
    mean_square_error = mse(original_image, reconstructed_image);
    max_pixel_value = 255; 
    psnr_value = 10 * log10((max_pixel_value^2) / mean_square_error);
end

function entropy = bitrate(image)
 
    flattened = round(image(:)); 
    min_val = min(flattened);    
   
    if min_val < 0
        flattened = flattened - min_val + 1;
    else
        flattened = flattened + 1;
    end

    max_val = max(flattened); 
    pixel_frequencies = accumarray(flattened, 1, [max_val, 1]);
    pixel_probabilities = pixel_frequencies / numel(flattened);
    non_zero_probabilities = pixel_probabilities(pixel_probabilities > 0);
    entropy = -sum(non_zero_probabilities .* log2(non_zero_probabilities));
end

function output = mse(img1, img2)
    % Mean Squared Error (MSE) calculation
    [M, N] = size(img1);
    output = sum((img1(:) - img2(:)).^2) / (M * N);
end
