

clear;
clc;

addpath('../lafmm/matlab');


N = 10^3;
m = 100;

k = 10;

% random rank k matrix plus noise
K = randn(N,k) * randn(k,m) + 0.01 * randn(N,m);


[~, R, perm] = qr(K,0);

[proj, skel] = InterpolativeDecompositionQR(R, perm, k);


K_approx = K(:,skel) * proj;

f_error = norm((K - K_approx), 'fro') / norm(K, 'fro');

fprintf('F-norm error: %f\n', f_error);


save('id_test_mat.out', 'K', '-ascii');

