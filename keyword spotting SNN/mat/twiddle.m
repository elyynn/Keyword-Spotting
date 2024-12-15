function generateApproxTwiddleFactors()
    % Define the number of points for FFT
    N = 512;

    % Generate the twiddle factors
    W = exp(-2j * pi * (0:N/2) / N)';

    % Separate the real and imaginary parts of the complex number
    real_part = real(W);
    imag_part = imag(W);

    % Find the closest values for both parts
    closest_real = findClosestValue(real_part);
    closest_imag = findClosestValue(imag_part);

    % Combine the results into a complex number
    twiddle_factors = complex(closest_real, closest_imag);
   
    save('twiddle_factors.mat', 'twiddle_factors');
end

function closest_values = findClosestValue(matrix)
    % Define the set of values to compare against
    % values = [-1, -0.5, 0, 0.5, 1];
    % values = [-1, -0.875, -0.75, -0.635, -0.5,-0.375 -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875,1];
     values = [-1, -0.9375, -0.875, -0.8125, -0.75, -0.6875, -0.625, -0.5625, -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625, 0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1]; 
     % Initialize the output matrix
    closest_values = zeros(size(matrix));
    
    % Process each element in the input matrix
    for i = 1:numel(matrix)
        x = matrix(i);
        [~, idx] = min(abs(values - x));
        closest_values(i) = values(idx);
    end
end

% Call the function to generate and save the approximate twiddle factors
generateApproxTwiddleFactors();