function maintrain()
    output_matfile = 'train_output.mat';
    input_matfile = 'TIDIGIT_train.mat';
    % Load the .mat file
    data = load(input_matfile);
    train_labels = data.train_labels;
    train_samples = data.train_samples;
    fs = 20000;

    % Initialize output cell arrays
    encode_labels = train_labels;
    encode_samples = cell(size(train_samples));

    % Process each sample
    for k = 1:length(train_samples)
        audio = train_samples{k};
        pre_emphasis_alpha = 0.97;
        pre_emphasized_audio = pre_emphasize(audio, pre_emphasis_alpha);

        frame_length_ms = 25;
        frame_shift_ms = 15;
        frame_length = round(frame_length_ms * fs / 1000);
        frame_shift = round(frame_shift_ms * fs / 1000);
        frames = frame_signal(pre_emphasized_audio, frame_length, frame_shift);

        window = custom_hamming_window(frame_length);
        windowed_frames = frames .* repmat(window, 1, size(frames, 2));

        FFT_frames = universal_fft(windowed_frames);
        power_spectrum = abs(FFT_frames).^2;

        num_filters = 20;
        mel_filter_bank = compute_mel_filter_bank(size(power_spectrum, 1), fs, num_filters);
        mel_spectrum = mel_filter_bank * power_spectrum;

        epsilon = 1;
        log_mel_spectrum = log(mel_spectrum + epsilon);

        [num_filters, num_frames] = size(log_mel_spectrum);
        normalized_log_mel_spectrum = zeros(num_filters, num_frames);
        for i = 1:num_filters
            min_val = min(log_mel_spectrum(i, :));
            max_val = max(log_mel_spectrum(i, :));
            normalized_log_mel_spectrum(i, :) = (log_mel_spectrum(i, :) - min_val) / (max_val - min_val);
        end

        thresholds = 0.0625 * (1:15);
        max_threshold = 1;
        num_features = size(normalized_log_mel_spectrum, 1);
        num_neurons = 2 * length(thresholds) * num_features + num_features;
        pulses = zeros(num_frames, num_neurons);

        for frame_idx = 1:num_frames
            frame_data = normalized_log_mel_spectrum(:, frame_idx);
            neuron_idx = 1;
            for feature_idx = 1:num_features
                for threshold_idx = 1:length(thresholds)
                    threshold = thresholds(threshold_idx);
                    if frame_idx > 1 && frame_data(feature_idx) > threshold && normalized_log_mel_spectrum(feature_idx, frame_idx-1) <= threshold
                        pulses(frame_idx, neuron_idx) = 1;
                    end
                    neuron_idx = neuron_idx + 1;
                    if frame_idx > 1 && frame_data(feature_idx) < threshold && normalized_log_mel_spectrum(feature_idx, frame_idx-1) >= threshold
                        pulses(frame_idx, neuron_idx) = 1;
                    end
                    neuron_idx = neuron_idx + 1;
                end
                if frame_data(feature_idx) >= max_threshold
                    pulses(frame_idx, neuron_idx) = 1;
                end
                neuron_idx = neuron_idx + 1;
            end
        end

        encoded_sample = pulses(:, 1:620);
        encode_samples{k} = encoded_sample;
    end

    % Save the encoded data to a .mat file
    save(output_matfile, 'encode_labels', 'encode_samples', 'fs');
    disp(['Encoded data saved to ', output_matfile]);
end

function y = pre_emphasize(signal, alpha)
    y = filter([1 -alpha], 1, signal);
end

function frames = frame_signal(signal, frame_length, frame_shift)
    num_frames = floor((length(signal) - frame_length) / frame_shift) + 1;
    frames = zeros(frame_length, num_frames);
    for i = 1:num_frames
        start_index = (i-1) * frame_shift + 1;
        frames(:, i) = signal(start_index:start_index + frame_length - 1);
    end
end

function w = custom_hamming_window(N)
    n = (0:N-1)';
    w = 0.54 - 0.46 * cos(2 * pi * n / (N-1));
end


function FFT_frames = universal_fft(frames)
    N = size(frames, 1);
    FFT_frames = zeros(N, size(frames, 2));
    for i = 1:size(frames, 2)
        FFT_frames(:, i) = fft(frames(:, i));
    end    
end


function mel_filter_bank = compute_mel_filter_bank(num_fft_bins, fs, num_filters)
    low_freq_mel = 0;
    high_freq_mel = 2595 * log10(1 + (fs / 2) / 700);
    mel_points = linspace(low_freq_mel, high_freq_mel, num_filters + 2);
    hz_points = 700 * (10.^(mel_points / 2595) - 1);
    bin = floor((num_fft_bins + 1) * hz_points / fs);
    mel_filter_bank = zeros(num_filters, num_fft_bins);
    for m = 2:(num_filters + 1)
        if bin(m-1) < 1 || bin(m) > num_fft_bins
            continue;
        end
        for k = bin(m-1):bin(m)
            mel_filter_bank(m-1, k) = (k - bin(m-1)) / (bin(m) - bin(m-1));
        end
        for k = bin(m):bin(m+1)
            if bin(m+1) > num_fft_bins
                continue;
            end
            mel_filter_bank(m-1, k) = (bin(m+1) - k) / (bin(m+1) - bin(m));
        end
    end
end