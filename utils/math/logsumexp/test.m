rng(5)

disp('Errors should be of order 1e-14.')

n = 100;
x = randn(n,1);

[sm1,lse1] = softmax_a(x);
[lse2,sm2] = logsumexp_a(x);
if any(sm2) < 0, error('Negative softmax element.'), end

error_1 = abs(sum(sm1) - 1)
error_2 = abs(sum(sm2) - 1)
error_3 = norm(sm1 - sm2,1)
error_4 = norm(lse1 - lse2,1)

sm3 = softmax_a(x);
error_5 = abs(sum(sm3) - 1)
if any(sm3) < 0, error('Negative softmax element.'), end

x(1:n-1) = -1e3;  % These elements underflow.
lse4 = logsumexp_a(x);
error_6 = lse4 - x(n)

x = [1e3 1 1];
lse5 = logsumexp_a(x);
if ~isfinite(lse5), error('Overflow.'), end
sm5 = softmax_a(x);
if ~isfinite(sm5), error('Overflow.'), end
error_7 = abs(sum(sm5) - 1)
if any(sm5) < 0, error('Negative softmax element.'), end

x = zeros(n,1);
[sm6,lse6] = softmax_a(x);
error_8 = lse6 - log(n)
error_9 = norm(sm6 - 1/n,1)

rnd = randn;
x = rnd*ones(n,1);
[sm7,lse7] = softmax_a(x);
error_10 = lse7 - log(n*exp(rnd))
error_11 = norm(sm7 - 1/n,1)
