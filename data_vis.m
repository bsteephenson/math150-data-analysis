knn_concat = csvread('./More_results_knn/knn_concatenate.csv');
knn_size = csvread('./More_results_knn/training_size_knn.csv');

neural_concat = csvread('./More_results_nn/neural_concatenate.csv');
neural_size = csvread('./More_results_nn/training_size_nn.csv');

% k = knn_concat(:,1);
% d = knn_concat(:,2);
% acc = knn_concat(:,13);

% Example to produce 3D plot
% scatter3(neural_concat(:,1),neural_concat(:,2),neural_concat(:,13),40, neural_concat(:,13),'filled');
% xlabel('Number of layers');
% ylabel('Number of nodes per layer');