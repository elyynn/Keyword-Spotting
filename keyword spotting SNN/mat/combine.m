% CombineTrainAndTest.m

% Load the train and test MAT files
train_data = load('train_output.mat');
test_data = load('test_output.mat');

% Extract the labels and patterns
train_labels = double(cell2mat(train_data.encode_labels));
train_pattern = train_data.encode_samples;
test_labels = double(cell2mat(test_data.encode_labels));
test_pattern = test_data.encode_samples;

% Save the combined data into a new MAT file in v7 format
save('combined_data.mat', 'train_labels', 'train_pattern', 'test_labels', 'test_pattern', '-v7');

disp('Combined MAT file created successfully.');