clc; clear; close all;

% --------- Image Loading ---------
images = {
    im2double(imread("images/boats512x512.tif")), 
    im2double(imread("images/harbour512x512.tif")), 
    im2double(imread("images/peppers512x512.tif"))
};
image_names = ["Boats", "Harbour", "Peppers"];

% --------- 2.1 Blockwise 8×8 DCT ---------
M = 8; 
A = generate_dct_matrix(M); % Generate DCT Transform Matrix
disp('DCT Transform Matrix A:');
disp(A);

% Test DCT and Inverse DCT on a single 8×8 block
boat = images{1}; 

%boat8x8 = imresize(images{1}, [8, 8]); 

% Visualize original, DCT coefficients, and reconstructed block
figure('Name', '2.1 Blockwise 8x8 DCT');
subplot(1, 3, 1);
imagesc(boat); colormap(gray); axis square; axis off;
title('Original 8×8 Block');

% Apply DCT transform
%dct_test = dct2_block(boat8x8, A);
dct_coef = blockproc(boat,[8 8],dct2_block(A));
subplot(1, 3, 2);
imagesc(dct_coef); colormap(gray); axis square; axis off;
title('DCT Coefficients');

% Apply Inverse DCT transform
%reconstructed_test = idct2_block(A);
reconstructed_test = blockproc(dct_coef,[8 8],idct2_block(A));
subplot(1, 3, 3);
imshow(reconstructed_test); colormap(gray); axis square; axis off;
title('Reconstructed Block');


% --------- 2.2 Uniform Quantizer ---------
figure('Name', '2.2 Uniform Quantizer');

x = 0:0.1:64; 
step_sizes = [2, 4, 8]; % Test different step sizes

hold on;
for step_size = step_sizes
    output = uniform_quantizer(x, step_size);
    plot(x, output, 'DisplayName', sprintf('Step Size = %d', step_size));
end
hold off;

title('Quantizer Function');
xlabel('Input');
ylabel('Quantized Output');
legend('show', 'Location', 'northwest');
grid on;

% --------- Supporting Functions ---------

% Function to generate DCT transform matrix
function A = generate_dct_matrix(M)
    A = zeros(M);
    for i = 0:M-1
        for j = 0:M-1
            if i == 0
                alpha = sqrt(1/M);
            else
                alpha = sqrt(2/M);
            end
            A(i+1, j+1) = alpha * cos((2*j + 1) * i * pi / (2 * M));
        end
    end
end

% Function to perform DCT using the matrix A
function  dct_coeffs = dct2_block(A)
    dct_coeffs = @(block_struct) A * block_struct.data * A';
end

% Function to perform inverse DCT using the matrix A
function reconstructed = idct2_block(A)
    reconstructed = @(block_struct) A' * block_struct.data * A;
end

% Uniform quantizer function
function quantized_output = uniform_quantizer(input, step_size)
    quantized_output = step_size * round(input / step_size);
end



% --------- 2.3 Distortion and Bit-Rate Estimation ---------


function d = MSE(original,reconstructed)
    [rows, columns, numberOfColorChannels] = size(reconstructed);
    d=mean((original(:) - reconstructed(:)).^2);
end 

d=MSE(boat,reconstructed_test)
d_original_to_coeff=MSE(boat,dct_coef)
PSNR= 10*log10(255^2/d)

