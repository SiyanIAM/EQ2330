clc;
clear;

load coeffs.mat
org_img = imread("images/harbour512x512.tif");
num_scales = 4;

% perform fwt
[approx_coeffs, hori_coeffs, verti_coeffs, diag_coeffs] = fwt(org_img, num_scales);

% Plot approximation coefficients
figure;
imagesc(approx_coeffs{num_scales});
title("Approximation Coefficients");
colormap gray(256);

% Plot horizontal coefficients
figure;
imagesc(hori_coeffs{num_scales});
title("Horizontal Coefficients");
colormap gray(256);

% Plot vertical coefficients
figure;
imagesc(verti_coeffs{num_scales});
title("Vertical Coefficients");
colormap gray(256);

% Plot diagonal coefficients
figure;
imagesc(diag_coeffs{num_scales});
title("Diagonal Coefficients");
colormap gray(256);



% 3.3 Uniform Quantizer
% Define quantizer function
step = 1;
quantizer = @(x) step * round(x / step);

% Quantize all coefficients directly in the main script
quantized_approx = cell(1, num_scales);
quantized_hori = cell(1, num_scales);
quantized_vert = cell(1, num_scales);
quantized_diag = cell(1, num_scales);

for i = 1:num_scales
    quantized_approx{i} = quantizer(approx_coeffs{i});
    quantized_hori{i} = quantizer(hori_coeffs{i});
    quantized_vert{i} = quantizer(verti_coeffs{i}); 
    quantized_diag{i} = quantizer(diag_coeffs{i});
end

% Reconstruct the image using quantized coefficients
rec_img = ifwt(quantized_approx, quantized_hori, quantized_vert, quantized_diag, num_scales);

% plot the image
figure;
imagesc(org_img);
title("Original Image");
colormap gray(256);

figure;
imagesc(rec_img);
title("Reconstructed Image");
colormap gray(256);

org_img = 255*im2double(org_img);

% calculate the mse between original image and reconstructed image
ori_re_mse = mse(org_img, rec_img);

disp("d between original and reconstructed: " + ori_re_mse);

% calculate the mse between each of the coefficients
% approx_mse = zeros(1,4);
% hori_mse = zeros(1,4);
% verti_mse = zeros(1,4);
% diag_mse = zeros(1,4);
% 
% for i = 1:num_scales
%     approx_mse(i) = mse(approx_coeffs{i},quantized_approx{i});
%     hori_mse(i) = mse(hori_coeffs{i},quantized_hori{i});
%     verti_mse(i) = mse(verti_coeffs{i},quantized_vert{i});
%     diag_mse(i) = mse(diag_coeffs{i},quantized_diag{i});
% end
% 
% % take the weighted average
% weighted_mse = 0;
% 
% for scale = 1:num_scales
%     weight = 1 / (4^scale);
%     weighted_mse = weighted_mse + (hori_mse(scale)+verti_mse(scale)+diag_mse(scale))* weight ;
% 
% end
% 
% weighted_mse = weighted_mse + approx_mse(num_scales)*(1/(4^num_scales));

function [approx_coeffs, hori_coeffs, verti_coeffs, diag_coeffs] = fwt(img, num_scales)
    % Daubechies 8-tap filter is used
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

        % First, work on the horizontal direction
        for x = 1:img_size
            [lowpass_temp(x, :), highpass_temp(x, :)] = anal_filter(current_image(x, :), LPF, HPF);
        end

        approx = zeros([img_size / 2, img_size / 2]);
        hori = zeros([img_size / 2, img_size / 2]);
        verti = zeros([img_size / 2, img_size / 2]);
        diag = zeros([img_size / 2, img_size / 2]);

        % Then, work on the vertical direction
        for y = 1:img_size / 2
            [approx(:, y), hori(:, y)] = anal_filter(lowpass_temp(:, y).', LPF, HPF);
            [verti(:, y), diag(:, y)] = anal_filter(highpass_temp(:, y).', LPF, HPF);
        end

        approx_coeffs{scale} = approx;
        hori_coeffs{scale} = hori;
        verti_coeffs{scale} = verti;
        diag_coeffs{scale} = diag;
        current_image = approx; % Update for the next scale
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
    % Daubechies 8-tap filter is used
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


function output = mse(img1, img2)
    % Mean Squared Error (MSE) calculation
    [M, N] = size(img1);
    output = sum((img1(:) - img2(:)).^2) / (M * N);
end